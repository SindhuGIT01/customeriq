"""Load the raw churn dataset, clean it, and save the processed version."""

import pandas as pd

from src.data.cleaning import clean_data

RAW_PATH = "data/raw/customers_raw.csv"
PROCESSED_PATH = "data/processed/customers_clean.csv"


def main():
    raw_df = pd.read_csv(RAW_PATH)
    print(f"Raw rows:     {len(raw_df)}")

    missing_before = raw_df.isna().sum()
    print("\nMissing values (before cleaning):")
    print(missing_before[missing_before > 0].to_string())

    clean_df = clean_data(raw_df)
    print(f"\nClean rows:   {len(clean_df)}")
    print(f"Rows removed: {len(raw_df) - len(clean_df)}")

    missing_after = clean_df.isna().sum()
    print("\nMissing values (after cleaning):")
    if missing_after.sum() == 0:
        print("none")
    else:
        print(missing_after[missing_after > 0].to_string())

    for col in ["monthly_spend_outlier", "tenure_months_outlier"]:
        print(f"\n{col}: {clean_df[col].sum()} rows flagged")

    clean_df.to_csv(PROCESSED_PATH, index=False)
    print(f"\nSaved cleaned data to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
