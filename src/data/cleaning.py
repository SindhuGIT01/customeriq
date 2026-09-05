"""Cleaning utilities for the raw CustomerIQ churn dataset."""

import pandas as pd

NUMERIC_COLUMNS = [
    "age", "tenure_months", "monthly_spend", "total_spend",
    "login_frequency", "support_tickets", "product_usage",
    "customer_lifetime_value",
]
CATEGORICAL_COLUMNS = ["gender", "location", "subscription_type", "payment_method"]
IQR_COLUMNS = ["monthly_spend", "tenure_months"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Fix data types -------------------------------------------------
    df["customer_id"] = df["customer_id"].astype(str).str.strip()

    # Raw exports sometimes carry stray text ("unknown") in numeric columns,
    # which forces pandas to load them as object/str. Coerce back to numeric,
    # turning any unparsable text into NaN so it's handled by the missing
    # value rules below rather than silently kept as a string.
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")

    # discount_used arrives as "Yes"/"No" text; a boolean is the natural type.
    df["discount_used"] = df["discount_used"].map({"Yes": True, "No": False})

    # --- Remove duplicate customer_id rows -------------------------------
    df = df.drop_duplicates(subset="customer_id", keep="first")

    # --- Handle missing values -------------------------------------------
    # age: roughly symmetric distribution -> median is a robust, unbiased fill.
    df["age"] = df["age"].fillna(df["age"].median())

    # location: nominal category with no natural "typical" value; an explicit
    # "Unknown" bucket avoids fabricating a customer's location.
    df["location"] = df["location"].cat.add_categories(["Unknown"]).fillna("Unknown")

    # tenure_months: business metric, skewed distribution -> median is more
    # robust to the long tail than the mean.
    df["tenure_months"] = df["tenure_months"].fillna(df["tenure_months"].median())

    # monthly_spend: also skewed by high-tier plans -> median again.
    df["monthly_spend"] = df["monthly_spend"].fillna(df["monthly_spend"].median())

    # total_spend: we have a domain formula (tenure * monthly rate), so
    # reconstruct missing values from it instead of guessing statistically.
    reconstructed_total = df["tenure_months"] * df["monthly_spend"]
    df["total_spend"] = df["total_spend"].fillna(reconstructed_total)

    # login_frequency: continuous usage metric -> median fill.
    df["login_frequency"] = df["login_frequency"].fillna(df["login_frequency"].median())

    # support_tickets: missing almost certainly means no ticket was ever
    # logged for that customer, not a random gap -> fill with 0.
    df["support_tickets"] = df["support_tickets"].fillna(0)

    # product_usage: continuous engagement score -> median fill.
    df["product_usage"] = df["product_usage"].fillna(df["product_usage"].median())

    # subscription_type: categorical -> fill with the most common plan, the
    # single best guess with no other signal available.
    mode_subscription = df["subscription_type"].mode(dropna=True)
    if not mode_subscription.empty:
        df["subscription_type"] = df["subscription_type"].cat.add_categories(
            [c for c in [mode_subscription.iloc[0]] if c not in df["subscription_type"].cat.categories]
        )
        df["subscription_type"] = df["subscription_type"].fillna(mode_subscription.iloc[0])

    # discount_used: conservative default is "no discount" when unrecorded,
    # since discounts are opt-in and absence of a record implies none applied.
    # NaN can't live in a true bool dtype column, which is why the earlier
    # map() left this as object dtype — cast for real only now that every
    # value is filled.
    df["discount_used"] = df["discount_used"].fillna(False).astype(bool)

    # churn: this is the prediction target. Imputing it would inject fake
    # labels into the training signal, so rows with an unknown outcome are
    # dropped instead of filled.
    df = df.dropna(subset=["churn"])
    df["churn"] = df["churn"].astype(int)

    # customer_lifetime_value: no reliable formula fallback (depends on
    # future behavior), so median fill for this remaining gap.
    df["customer_lifetime_value"] = df["customer_lifetime_value"].fillna(
        df["customer_lifetime_value"].median()
    )

    # --- Flag and handle outliers (IQR method) ---------------------------
    for col in IQR_COLUMNS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flag_col = f"{col}_outlier"
        df[flag_col] = (df[col] < lower) | (df[col] > upper)
        # Cap rather than drop: keeps the row (and its churn label) while
        # neutralizing the extreme value's influence on downstream stats/models.
        df[col] = df[col].clip(lower=lower, upper=upper)

    df["age"] = df["age"].astype(int)
    df["support_tickets"] = df["support_tickets"].astype(int)
    df["tenure_months"] = df["tenure_months"].round().astype(int)

    return df.reset_index(drop=True)
