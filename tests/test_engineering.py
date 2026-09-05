import numpy as np
import pandas as pd

from src.features.engineering import encode_and_scale, engineer_features

NEW_RATIO_COLUMNS = ["average_monthly_spend", "usage_per_month", "support_ticket_rate"]
NEW_GROUP_COLUMNS = ["age_group", "tenure_group"]


def _sample_df():
    return pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4"],
        "age": [22, 30, 45, 60],
        "gender": ["Male", "Female", "Female", "Other"],
        "location": ["Austin", "Dallas", "Austin", "Chicago"],
        "tenure_months": [0, 5, 15, 30],
        "monthly_spend": [20.0, 45.0, 80.0, 150.0],
        "total_spend": [0.0, 225.0, 1200.0, 4500.0],
        "login_frequency": [10.0, 12.0, 8.0, 20.0],
        "support_tickets": [0, 1, 3, 0],
        "product_usage": [50.0, 60.0, 40.0, 90.0],
        "subscription_type": ["Basic", "Standard", "Premium", "Enterprise"],
        "payment_method": ["PayPal", "Credit Card", "Debit Card", "Bank Transfer"],
        "discount_used": [False, True, False, True],
        "churn": [0, 1, 0, 0],
        "customer_lifetime_value": [500.0, 1200.0, 3000.0, 9000.0],
        "monthly_spend_outlier": [False, False, False, False],
        "tenure_months_outlier": [False, False, False, False],
    })


def test_engineer_features_adds_expected_columns_with_no_nan_or_inf():
    df = engineer_features(_sample_df())

    for col in NEW_RATIO_COLUMNS + NEW_GROUP_COLUMNS:
        assert col in df.columns
        assert not df[col].isna().any()

    ratio_values = df[NEW_RATIO_COLUMNS].to_numpy(dtype=float)
    assert np.isfinite(ratio_values).all()


def test_engineer_features_guards_zero_tenure():
    df = engineer_features(_sample_df())
    zero_tenure_row = df.loc[df["tenure_months"] == 0].iloc[0]

    for col in NEW_RATIO_COLUMNS:
        assert np.isfinite(zero_tenure_row[col])


def test_encode_and_scale_has_no_nan_or_inf():
    df = engineer_features(_sample_df())
    transformed, _encoder, _scaler = encode_and_scale(df)

    assert not transformed.isna().any().any()
    numeric_block = transformed.select_dtypes(include=[np.number]).to_numpy()
    assert np.isfinite(numeric_block).all()


def test_encode_and_scale_reuses_fitted_encoder_and_scaler():
    df = engineer_features(_sample_df())
    _, encoder, scaler = encode_and_scale(df)

    transformed_again, _, _ = encode_and_scale(df, encoder=encoder, scaler=scaler)

    assert not transformed_again.isna().any().any()
    numeric_block = transformed_again.select_dtypes(include=[np.number]).to_numpy()
    assert np.isfinite(numeric_block).all()
