import numpy as np
import pandas as pd

from src.data.cleaning import clean_data


def _dirty_df():
    return pd.DataFrame({
        "customer_id": ["C1", "C2", "C2", "C3", "C4", "C5"],  # C2 duplicated
        "age": [25, np.nan, 40, 60, 35, 45],
        "gender": ["Male", "Female", "Female", "Male", None, "Other"],
        "location": ["Austin", None, "Austin", "Dallas", "Chicago", "Chicago"],
        "tenure_months": ["12", "unknown", "5", "500", "8", "20"],  # "unknown" + outlier
        "monthly_spend": [20.0, 45.0, 45.0, 5000.0, 30.0, np.nan],  # outlier + nan
        "total_spend": [240.0, np.nan, 225.0, 25000.0, 240.0, 600.0],
        "login_frequency": [10.0, 12.0, 12.0, 5.0, 8.0, 15.0],
        "support_tickets": [0, 1, 1, np.nan, 2, 0],
        "product_usage": [50.0, 60.0, 60.0, 40.0, np.nan, 70.0],
        "subscription_type": ["Basic", "Standard", "Standard", None, "Premium", "Enterprise"],
        "payment_method": ["PayPal", "Credit Card", "Credit Card", "Debit Card", "PayPal", "Bank Transfer"],
        "discount_used": ["Yes", "No", "No", "Yes", None, "No"],
        "churn": [0, 1, 1, 0, np.nan, 0],  # one missing target
        "customer_lifetime_value": [500.0, 1200.0, 1200.0, 9000.0, np.nan, 2500.0],
    })


def test_clean_data_removes_duplicate_customer_ids():
    result = clean_data(_dirty_df())
    assert result["customer_id"].duplicated().sum() == 0


def test_clean_data_drops_rows_with_missing_churn():
    result = clean_data(_dirty_df())
    assert "C4" not in result["customer_id"].values


def test_clean_data_leaves_no_missing_values():
    result = clean_data(_dirty_df())
    assert result.isna().sum().sum() == 0


def test_clean_data_fixes_dtypes():
    result = clean_data(_dirty_df())
    assert pd.api.types.is_integer_dtype(result["tenure_months"])
    assert pd.api.types.is_integer_dtype(result["age"])
    assert pd.api.types.is_bool_dtype(result["discount_used"])


def test_clean_data_flags_outliers():
    result = clean_data(_dirty_df())
    assert "monthly_spend_outlier" in result.columns
    assert "tenure_months_outlier" in result.columns
    assert bool(result["monthly_spend_outlier"].any())
