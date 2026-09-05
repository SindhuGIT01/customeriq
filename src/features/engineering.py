"""Feature engineering for the CustomerIQ churn dataset."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLUMNS = [
    "gender", "location", "subscription_type", "payment_method",
    "age_group", "tenure_group",
]
NUMERIC_COLUMNS = [
    "age", "tenure_months", "monthly_spend", "total_spend",
    "login_frequency", "support_tickets", "product_usage",
    "customer_lifetime_value", "average_monthly_spend",
    "usage_per_month", "support_ticket_rate",
]

AGE_BINS = [-np.inf, 24, 34, 44, 54, np.inf]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55+"]

TENURE_BINS = [-np.inf, 6, 12, 24, np.inf]
TENURE_LABELS = ["0-6mo", "6-12mo", "1-2yr", "2yr+"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived ratio and bucket features to the cleaned dataset."""
    df = df.copy()

    # Guard divide-by-zero for brand-new customers (tenure_months == 0) by
    # treating anything under one month as a 1-month base for rate math.
    tenure_denominator = df["tenure_months"].clip(lower=1)

    df["average_monthly_spend"] = df["total_spend"] / tenure_denominator
    df["usage_per_month"] = df["product_usage"] / tenure_denominator
    df["support_ticket_rate"] = df["support_tickets"] / tenure_denominator

    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS)
    df["tenure_group"] = pd.cut(df["tenure_months"], bins=TENURE_BINS, labels=TENURE_LABELS)

    return df


def encode_and_scale(
    df: pd.DataFrame,
    categorical_columns=CATEGORICAL_COLUMNS,
    numeric_columns=NUMERIC_COLUMNS,
    encoder: OneHotEncoder | None = None,
    scaler: StandardScaler | None = None,
):
    """One-hot encode categoricals and standard-scale numerics.

    Pass a previously-fitted `encoder`/`scaler` (as returned from an earlier
    call on the training set) to transform new data the same way at
    prediction time.
    """
    df = df.copy()
    categorical_data = df[categorical_columns].astype(str)

    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = encoder.fit_transform(categorical_data)
    else:
        encoded = encoder.transform(categorical_data)
    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(categorical_columns), index=df.index
    )

    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[numeric_columns])
    else:
        scaled = scaler.transform(df[numeric_columns])
    scaled_df = pd.DataFrame(scaled, columns=numeric_columns, index=df.index)

    passthrough_columns = [
        c for c in df.columns if c not in categorical_columns and c not in numeric_columns
    ]
    transformed = pd.concat([df[passthrough_columns], scaled_df, encoded_df], axis=1)
    return transformed, encoder, scaler
