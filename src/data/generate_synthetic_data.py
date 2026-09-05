"""Generate a synthetic telecom/SaaS customer churn dataset.

Real public churn datasets (e.g. IBM Telco Customer Churn) don't carry the
column schema this project needs (age, location, login_frequency,
support_tickets, product_usage, customer_lifetime_value, etc.), so we
generate synthetic data instead. Values are drawn so that:
  - total_spend roughly tracks tenure_months * monthly_spend
  - churn probability rises with support_tickets and falls with
    login_frequency / product_usage / tenure_months
Some missingness, duplicate customer_id rows, extreme outliers, and mixed
dtypes are injected on purpose so src/data/cleaning.py has real work to do.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_ROWS = 2500
OUTPUT_PATH = "data/raw/customers_raw.csv"

LOCATIONS = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "San Antonio", "San Diego", "Dallas", "Austin", "Remote/Other",
]
SUBSCRIPTION_TYPES = ["Basic", "Standard", "Premium", "Enterprise"]
SUBSCRIPTION_WEIGHTS = [0.35, 0.35, 0.20, 0.10]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "Bank Transfer", "Electronic Check"]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n_rows: int = N_ROWS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = [f"CUST{100000 + i}" for i in range(n_rows)]
    age = rng.normal(42, 13, n_rows).clip(18, 85).round().astype(int)
    gender = rng.choice(["Male", "Female", "Other"], size=n_rows, p=[0.48, 0.48, 0.04])
    location = rng.choice(LOCATIONS, size=n_rows)
    subscription_type = rng.choice(SUBSCRIPTION_TYPES, size=n_rows, p=SUBSCRIPTION_WEIGHTS)
    payment_method = rng.choice(PAYMENT_METHODS, size=n_rows)
    discount_used = rng.choice(["Yes", "No"], size=n_rows, p=[0.3, 0.7])

    # Tenure: many newer customers, a long tail of long-term ones.
    tenure_months = rng.exponential(scale=24, size=n_rows).clip(0, 72).round().astype(int)

    # Base monthly spend depends on subscription tier.
    tier_mean = {"Basic": 20, "Standard": 45, "Premium": 80, "Enterprise": 150}
    tier_std = {"Basic": 5, "Standard": 8, "Premium": 12, "Enterprise": 25}
    monthly_spend = np.array([
        rng.normal(tier_mean[t], tier_std[t]) for t in subscription_type
    ]).clip(5, None).round(2)

    # Engagement metrics: correlated with each other, will drive churn risk.
    product_usage = rng.normal(65, 15, n_rows).clip(0, 100)
    login_frequency = (product_usage / 100 * 25 + rng.normal(0, 3, n_rows)).clip(0, None).round(1)
    support_tickets = rng.poisson(lam=1.0, size=n_rows)
    # Low-usage customers tend to raise more tickets.
    support_tickets = (support_tickets + (product_usage < 35).astype(int) * rng.poisson(1.5, n_rows))

    total_spend = (tenure_months * monthly_spend * rng.normal(1.0, 0.05, n_rows)).clip(0, None).round(2)

    customer_lifetime_value = (
        monthly_spend * tenure_months * 1.5
        + product_usage * 10
        - support_tickets * 20
    ).clip(0, None).round(2)

    # Churn probability: down with tenure/usage/login, up with support tickets.
    z = (
        -0.04 * tenure_months
        + 0.35 * support_tickets
        - 0.03 * login_frequency
        - 0.02 * product_usage
        + 0.005 * monthly_spend
        - 1.0
    )
    churn_prob = _sigmoid(z)
    churn = rng.binomial(1, churn_prob)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "gender": gender,
        "location": location,
        "tenure_months": tenure_months,
        "monthly_spend": monthly_spend,
        "total_spend": total_spend,
        "login_frequency": login_frequency,
        "support_tickets": support_tickets,
        "product_usage": product_usage.round(1),
        "subscription_type": subscription_type,
        "payment_method": payment_method,
        "discount_used": discount_used,
        "churn": churn,
        "customer_lifetime_value": customer_lifetime_value,
    })

    df = _inject_messiness(df, rng)
    return df


def _inject_messiness(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Returns a dirtied copy: missing values, outliers, dup ids, mixed dtypes."""
    df = df.copy()
    n = len(df)

    missing_rates = {
        "age": 0.02,
        "location": 0.05,
        "monthly_spend": 0.01,
        "total_spend": 0.03,
        "support_tickets": 0.04,
        "product_usage": 0.03,
        "subscription_type": 0.02,
        "discount_used": 0.06,
        "customer_lifetime_value": 0.04,
    }
    for col, rate in missing_rates.items():
        mask = rng.random(n) < rate
        df.loc[mask, col] = np.nan

    # A few rows with missing target (should be dropped, not imputed).
    churn_missing_idx = rng.choice(n, size=10, replace=False)
    df.loc[churn_missing_idx, "churn"] = np.nan

    # Mixed dtypes: sprinkle a non-numeric sentinel into numeric columns
    # (not one of pandas' default NA strings) so the raw CSV loads these as
    # object dtype instead of clean numeric NaN.
    for col in ["tenure_months", "support_tickets"]:
        mask = rng.random(n) < 0.01
        df[col] = df[col].astype(object)
        df.loc[mask, col] = "unknown"

    # Outliers: extreme data-entry-error-style values.
    outlier_idx = rng.choice(n, size=max(1, int(n * 0.01)), replace=False)
    df.loc[outlier_idx, "monthly_spend"] = rng.uniform(3000, 6000, size=len(outlier_idx))
    outlier_idx2 = rng.choice(n, size=max(1, int(n * 0.01)), replace=False)
    df.loc[outlier_idx2, "tenure_months"] = rng.integers(200, 400, size=len(outlier_idx2))

    # Duplicate customer_id rows (simulating re-submitted records).
    dup_idx = rng.choice(n, size=max(1, int(n * 0.015)), replace=False)
    dup_rows = df.loc[dup_idx]
    df = pd.concat([df, dup_rows], ignore_index=True)

    # Shuffle so duplicates aren't all trailing the file.
    df = df.sample(frac=1, random_state=rng.integers(0, 2**32 - 1)).reset_index(drop=True)
    return df


def main():
    df = generate()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
