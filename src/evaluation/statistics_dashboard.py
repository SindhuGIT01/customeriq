"""Dataset statistics for the CustomerIQ dashboard/API.

compute_statistics() returns a plain dict of native Python types (no numpy
or pandas objects) so it can be serialized directly to JSON for an API
response.
"""

import json

import pandas as pd

NUMERIC_COLUMNS = [
    "age", "tenure_months", "monthly_spend", "total_spend",
    "login_frequency", "support_tickets", "product_usage",
    "customer_lifetime_value",
]
DATA_PATH = "data/processed/customers_clean.csv"


def compute_statistics(df: pd.DataFrame) -> dict:
    """Summarize the dataset: counts, per-column stats, missingness,
    duplicates, and the numeric correlation matrix."""
    numeric_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]

    summary_statistics = {
        col: {
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()),
        }
        for col in numeric_cols
    }

    correlation_matrix = df[numeric_cols].corr()
    correlation_dict = {
        row: {col: float(correlation_matrix.loc[row, col]) for col in numeric_cols}
        for row in numeric_cols
    }

    missing_value_counts = {
        col: int(count) for col, count in df.isna().sum().items() if count > 0
    }

    duplicate_customer_id_count = (
        int(df["customer_id"].duplicated().sum()) if "customer_id" in df.columns else None
    )
    churn_rate = float(df["churn"].mean()) if "churn" in df.columns else None

    return {
        "customer_count": len(df),
        "churn_rate": churn_rate,
        "summary_statistics": summary_statistics,
        "missing_value_counts": missing_value_counts,
        "duplicate_customer_id_count": duplicate_customer_id_count,
        "correlation_matrix": correlation_dict,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    stats = compute_statistics(df)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
