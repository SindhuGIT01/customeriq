"""Modular ETL pipeline for the CustomerIQ churn dataset.

extract -> validate -> transform -> save, wired together by run_etl().
"""

import argparse

import pandas as pd

from src.data.cleaning import clean_data

REQUIRED_COLUMNS = [
    "customer_id", "age", "gender", "location", "tenure_months",
    "monthly_spend", "total_spend", "login_frequency", "support_tickets",
    "product_usage", "subscription_type", "payment_method", "discount_used",
    "churn", "customer_lifetime_value",
]
NUMERIC_COLUMNS = [
    "age", "tenure_months", "monthly_spend", "total_spend",
    "login_frequency", "support_tickets", "product_usage",
    "customer_lifetime_value",
]
CATEGORICAL_TEXT_COLUMNS = ["gender", "location", "subscription_type", "payment_method"]
MAX_NON_NUMERIC_FRACTION = 0.5


def extract(path: str) -> pd.DataFrame:
    """Load the raw customer CSV."""
    return pd.read_csv(path)


def validate(df: pd.DataFrame) -> None:
    """Check schema and basic type sanity, raising clear errors on failure."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df["customer_id"].isna().all():
        raise ValueError("customer_id column is entirely empty")

    for col in NUMERIC_COLUMNS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        # Non-numeric text mixed into an otherwise numeric column (e.g. an
        # "unknown" sentinel) is expected and handled downstream; a column
        # that is mostly non-numeric text means the wrong column/schema.
        non_numeric = df[col].notna() & coerced.isna()
        non_numeric_fraction = non_numeric.mean() if len(df) else 0
        if non_numeric_fraction > MAX_NON_NUMERIC_FRACTION:
            raise TypeError(
                f"Column '{col}' expected to be numeric but "
                f"{non_numeric_fraction:.0%} of non-null values are not "
                "numeric-coercible"
            )

    churn_values = pd.to_numeric(df["churn"], errors="coerce").dropna().unique()
    invalid_churn = set(churn_values) - {0, 1}
    if invalid_churn:
        raise ValueError(f"churn column contains invalid values: {invalid_churn}")


def _standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Trim stray whitespace on free-text category columns.

    Deliberately does not force title/upper/lower casing: values like
    "PayPal" have meaningful internal capitalization that a blanket case
    transform would corrupt (e.g. .str.title() turns it into "Paypal").
    """
    df = df.copy()
    for col in CATEGORICAL_TEXT_COLUMNS:
        df[col] = df[col].astype(str).str.strip().astype("category")
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data, then apply extra type/category standardization."""
    df = clean_data(df)
    df = _standardize_categories(df)
    return df


def run_etl(input_path: str, output_path: str) -> pd.DataFrame:
    df = extract(input_path)
    validate(df)
    clean_df = transform(df)
    clean_df.to_csv(output_path, index=False)
    return clean_df


def main():
    parser = argparse.ArgumentParser(description="Run the CustomerIQ ETL pipeline.")
    parser.add_argument("--input", default="data/raw/customers_raw.csv")
    parser.add_argument("--output", default="data/processed/customers_clean.csv")
    args = parser.parse_args()

    clean_df = run_etl(args.input, args.output)
    print(f"ETL complete: {len(clean_df)} rows written to {args.output}")


if __name__ == "__main__":
    main()
