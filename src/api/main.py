"""FastAPI application exposing the CustomerIQ churn/CLV/segmentation pipeline.

Trained models are loaded once at startup with joblib (not retrained per
request). /train re-runs training and reloads them into memory.
"""

import io
import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.data.cleaning import clean_data
from src.data.preprocessing import validate
from src.evaluation.metrics import to_json_safe
from src.evaluation.statistics_dashboard import compute_statistics
from src.features.engineering import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    engineer_features,
)
from src.models.clustering import CLUSTER_FEATURES

RAW_PATH = "data/raw/customers_raw.csv"
PROCESSED_PATH = "data/processed/customers_clean.csv"
MODELS_DIR = "models"
CHURN_MODEL_PATH = os.path.join(MODELS_DIR, "best_churn_model.joblib")
CLUSTER_MODEL_PATH = os.path.join(MODELS_DIR, "clustering_model.joblib")
CLASSIFICATION_METRICS_PATH = os.path.join(MODELS_DIR, "classification_metrics.json")
REGRESSION_METRICS_PATH = os.path.join(MODELS_DIR, "regression_metrics.json")
CLUSTERING_METRICS_PATH = os.path.join(MODELS_DIR, "clustering_profile.json")

RAW_INPUT_COLUMNS = [
    "customer_id", "age", "gender", "location", "tenure_months",
    "monthly_spend", "total_spend", "login_frequency", "support_tickets",
    "product_usage", "subscription_type", "payment_method", "discount_used",
    "customer_lifetime_value",
]

# Plain-English (high, low) phrasing for continuous/rate features, keyed by
# their scaled value's sign (>= 0 means above the training-set average).
NUMERIC_FEATURE_LABELS = {
    "age": ("Older customer", "Younger customer"),
    "tenure_months": ("Long tenure", "Short tenure"),
    "monthly_spend": ("High monthly spend", "Low monthly spend"),
    "total_spend": ("High total spend", "Low total spend"),
    "login_frequency": ("Frequent logins", "Infrequent logins"),
    "support_tickets": ("Many support tickets", "Few support tickets"),
    "product_usage": ("High product usage", "Low product usage"),
    "customer_lifetime_value": ("High customer lifetime value", "Low customer lifetime value"),
    "average_monthly_spend": ("High average monthly spend", "Low average monthly spend"),
    "usage_per_month": ("High usage per month", "Low usage per month"),
    "support_ticket_rate": ("High support-ticket rate", "Low support-ticket rate"),
}
BOOLEAN_FEATURE_LABELS = {
    "discount_used": ("Uses a discount", "No discount applied"),
    "monthly_spend_outlier": ("Unusually extreme monthly spend", "Typical monthly spend"),
    "tenure_months_outlier": ("Unusually extreme tenure", "Typical tenure"),
}
TOP_RISK_FACTORS = 4
TOP_PROTECTIVE_FACTORS = 3

state: dict = {}


# --- Pydantic schemas --------------------------------------------------

class CustomerFeatures(BaseModel):
    """Raw customer feature values. Any field may be omitted; missing
    fields are filled from the training set's per-column median/mode."""
    age: float | None = None
    gender: str | None = None
    location: str | None = None
    tenure_months: float | None = None
    monthly_spend: float | None = None
    total_spend: float | None = None
    login_frequency: float | None = None
    support_tickets: float | None = None
    product_usage: float | None = None
    subscription_type: str | None = None
    payment_method: str | None = None
    discount_used: bool | None = None
    customer_lifetime_value: float | None = None


class SegmentRequest(BaseModel):
    customer_id: str | None = None
    features: CustomerFeatures | None = None


class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: str
    risk_factors: list[str]
    protective_factors: list[str]


class ExplanationBreakdown(BaseModel):
    risk_factors: list[str]
    protective_factors: list[str]


class ExplainResponse(BaseModel):
    customer_id: str
    churn_probability: float
    prediction: str
    coefficient_based: ExplanationBreakdown
    shap_based: ExplanationBreakdown | None = None
    shap_error: str | None = None


class SegmentResponse(BaseModel):
    cluster: int
    segment_label: str


class UploadResponse(BaseModel):
    rows_ingested: int
    rows_after_cleaning: int
    message: str


class TrainResponse(BaseModel):
    classification_metrics: list
    regression_metrics: list
    clustering_profile: list


class ModelMetricsResponse(BaseModel):
    classification: dict | None = None
    regression: dict | None = None
    clustering: dict | None = None


class StatisticsResponse(BaseModel):
    customer_count: int
    churn_rate: float | None
    summary_statistics: dict
    missing_value_counts: dict
    duplicate_customer_id_count: int | None
    correlation_matrix: dict


class CustomerRecordResponse(BaseModel):
    customer_id: str
    record: dict
    churn_probability: float | None = None
    prediction: str | None = None
    cluster: int | None = None
    segment_label: str | None = None


# --- State loading -------------------------------------------------------

def _compute_defaults(df: pd.DataFrame) -> dict:
    """Per-column median (numeric) / mode (categorical) used to fill in any
    field a prediction/segment request omits."""
    defaults = {}
    for col in RAW_INPUT_COLUMNS:
        if col == "customer_id" or col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            defaults[col] = bool(series.mode(dropna=True).iloc[0])
        elif pd.api.types.is_numeric_dtype(series):
            defaults[col] = float(series.median())
        else:
            defaults[col] = series.mode(dropna=True).iloc[0]
    return defaults


def load_state() -> None:
    """(Re)load the customer dataset and all trained model bundles into memory."""
    if os.path.exists(PROCESSED_PATH):
        df = pd.read_csv(PROCESSED_PATH)
        state["customers"] = df
        state["defaults"] = _compute_defaults(df)
    else:
        state["customers"] = pd.DataFrame()
        state["defaults"] = {}

    state["churn_bundle"] = joblib.load(CHURN_MODEL_PATH) if os.path.exists(CHURN_MODEL_PATH) else None
    state["cluster_bundle"] = joblib.load(CLUSTER_MODEL_PATH) if os.path.exists(CLUSTER_MODEL_PATH) else None
    # Invalidate any cached SHAP explainer — it was fit against the model
    # and data that just got replaced.
    state["shap_explainer"] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_state()
    yield


app = FastAPI(title="CustomerIQ API", lifespan=lifespan)


# --- Shared prediction helpers -------------------------------------------

def _merge_with_defaults(features: dict) -> dict:
    merged = dict(state["defaults"])
    merged.update({k: v for k, v in features.items() if v is not None})
    return merged


def _build_churn_feature_vector(feature_dict: dict) -> tuple:
    """Turn raw feature values into the exact row the churn model expects.
    Returns (X_row as a one-row DataFrame, model, feature_directions dict)."""
    bundle = state.get("churn_bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Classification model is not trained yet. Call /train first.")

    row = _merge_with_defaults(feature_dict)
    row_df = pd.DataFrame([row])
    row_df = engineer_features(row_df)

    model, encoder, scaler = bundle["model"], bundle["encoder"], bundle["scaler"]

    X = pd.DataFrame(0.0, index=[0], columns=list(model.feature_names_in_))

    scaled = scaler.transform(row_df[NUMERIC_COLUMNS])
    for i, col in enumerate(NUMERIC_COLUMNS):
        if col in X.columns:
            X.loc[0, col] = scaled[0, i]

    encoded = encoder.transform(row_df[CATEGORICAL_COLUMNS].astype(str))
    encoded_columns = encoder.get_feature_names_out(CATEGORICAL_COLUMNS)
    for i, col in enumerate(encoded_columns):
        if col in X.columns:
            X.loc[0, col] = encoded[0, i]

    if "discount_used" in X.columns:
        X.loc[0, "discount_used"] = float(bool(row.get("discount_used", False)))
    # monthly_spend_outlier / tenure_months_outlier are dataset-wide IQR
    # flags; not well-defined for a single new customer, so default to 0.

    return X, model, bundle.get("feature_directions", {})


def _predict_churn(feature_dict: dict) -> tuple:
    X, model, _ = _build_churn_feature_vector(feature_dict)
    proba = float(model.predict_proba(X)[0][1])
    prediction = "High Risk" if proba >= 0.5 else "Low Risk"
    return proba, prediction


def _describe_feature(feature_name: str, value: float) -> str:
    """Turn a raw (post-encoding) feature name + its value into a
    plain-English label, e.g. 'support_ticket_rate' -> 'High support-ticket
    rate' or 'subscription_type_Enterprise' -> 'Enterprise subscription'."""
    if feature_name in NUMERIC_FEATURE_LABELS:
        high_label, low_label = NUMERIC_FEATURE_LABELS[feature_name]
        return high_label if value >= 0 else low_label
    if feature_name in BOOLEAN_FEATURE_LABELS:
        true_label, false_label = BOOLEAN_FEATURE_LABELS[feature_name]
        return true_label if value >= 0.5 else false_label
    for cat_col in CATEGORICAL_COLUMNS:
        prefix = f"{cat_col}_"
        if feature_name.startswith(prefix):
            category = feature_name[len(prefix):]
            pretty_col = cat_col.replace("_", " ")
            return f"{category} {pretty_col}"
    return feature_name.replace("_", " ")


def _factors_from_contributions(contributions: pd.Series, X_row: pd.Series) -> dict:
    """Rank signed per-feature contributions into risk (positive, pushes
    toward churn) and protective (negative, pushes away) plain-English
    factors for one specific prediction."""
    contributions = contributions[contributions.abs() > 1e-9]
    risk = contributions[contributions > 0].sort_values(ascending=False).head(TOP_RISK_FACTORS)
    protective = contributions[contributions < 0].sort_values().head(TOP_PROTECTIVE_FACTORS)
    return {
        "risk_factors": [_describe_feature(name, X_row[name]) for name in risk.index],
        "protective_factors": [_describe_feature(name, X_row[name]) for name in protective.index],
    }


def _explain_churn_coefficients(feature_dict: dict) -> dict:
    """Explain a prediction using the model's own coefficients (Logistic
    Regression) or feature_importances_ (tree-based models).

    Logistic Regression: each feature's contribution to the log-odds of
    churn is exactly coefficient * scaled_value, so the sign tells us
    directly whether that feature pushed this customer toward or away from
    churn, and the magnitude ranks how much.

    Tree-based models: feature_importances_ only says how much a feature
    matters *on average across the whole forest*, not which direction it
    points for a specific customer. We approximate a direction by
    multiplying importance by the feature's correlation sign with churn in
    the training data (precomputed and saved with the model) and by this
    customer's own scaled value — a coarse stand-in for a real per-instance
    attribution (see the SHAP comparison below).
    """
    X, model, feature_directions = _build_churn_feature_vector(feature_dict)
    X_row = X.iloc[0]

    if hasattr(model, "coef_"):
        contributions = pd.Series(model.coef_[0] * X_row.to_numpy(), index=X_row.index)
    elif hasattr(model, "feature_importances_"):
        directions = pd.Series(feature_directions).reindex(X_row.index).fillna(0.0)
        contributions = pd.Series(
            model.feature_importances_ * directions.to_numpy() * X_row.to_numpy(), index=X_row.index
        )
    else:
        raise HTTPException(status_code=500, detail="Model has no coefficients or feature importances")

    return _factors_from_contributions(contributions, X_row)


def _get_shap_explainer():
    """Build (once) and cache a SHAP explainer for the current churn model,
    fit against a background sample of the current customer dataset."""
    if state.get("shap_explainer") is not None:
        return state["shap_explainer"]

    import shap

    bundle = state.get("churn_bundle")
    customers = state.get("customers", pd.DataFrame())
    if bundle is None or customers.empty:
        return None

    from src.features.engineering import encode_and_scale

    model, encoder, scaler = bundle["model"], bundle["encoder"], bundle["scaler"]
    df = engineer_features(customers.copy())
    transformed, _, _ = encode_and_scale(df, encoder=encoder, scaler=scaler)
    X_background = transformed.reindex(columns=model.feature_names_in_, fill_value=0)
    background_sample = X_background.sample(min(100, len(X_background)), random_state=42)

    explainer = shap.LinearExplainer(model, background_sample) if hasattr(model, "coef_") \
        else shap.TreeExplainer(model)
    state["shap_explainer"] = explainer
    return explainer


def _explain_churn_shap(feature_dict: dict) -> dict:
    """Explain a prediction using SHAP (Shapley additive explanations)."""
    X, _model, _directions = _build_churn_feature_vector(feature_dict)
    X_row = X.iloc[0]

    explainer = _get_shap_explainer()
    if explainer is None:
        raise ValueError("Not enough data to build a SHAP background sample. Call /upload first.")

    raw_values = explainer.shap_values(X)
    # LinearExplainer returns (1, n_features); TreeExplainer on a binary
    # classifier returns (1, n_features, n_classes) — take the churn class.
    values = raw_values[0, :, 1] if raw_values.ndim == 3 else raw_values[0]
    contributions = pd.Series(values, index=X_row.index)

    return _factors_from_contributions(contributions, X_row)


def _predict_segment(feature_dict: dict) -> tuple:
    bundle = state.get("cluster_bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Clustering model is not trained yet. Call /train first.")

    row = _merge_with_defaults(feature_dict)
    row_df = pd.DataFrame([row])
    row_df = engineer_features(row_df)

    kmeans, scaler, segment_labels = bundle["kmeans"], bundle["scaler"], bundle["segment_labels"]
    X_scaled = scaler.transform(row_df[CLUSTER_FEATURES])
    cluster = int(kmeans.predict(X_scaled)[0])
    return cluster, segment_labels[cluster]


def _customer_row(customer_id: str) -> dict:
    customers = state.get("customers", pd.DataFrame())
    if customers.empty or "customer_id" not in customers.columns:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found")
    matches = customers[customers["customer_id"] == customer_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found")
    return to_json_safe(matches.iloc[0].to_dict())


# --- Endpoints -------------------------------------------------------------

@app.post("/upload", response_model=UploadResponse)
async def upload_customers(file: UploadFile = File(...)):  # noqa: B008 (standard FastAPI idiom)
    contents = await file.read()
    try:
        raw_df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}") from e

    try:
        validate(raw_df)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    raw_df.to_csv(RAW_PATH, index=False)

    clean_df = clean_data(raw_df)
    engineer_features(clean_df)  # validates the feature-engineering step runs cleanly

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    clean_df.to_csv(PROCESSED_PATH, index=False)

    load_state()

    return UploadResponse(
        rows_ingested=len(raw_df),
        rows_after_cleaning=len(clean_df),
        message="Data cleaned, feature-engineering validated, and stored as the current processed dataset.",
    )


@app.post("/train", response_model=TrainResponse)
def train_models():
    if not os.path.exists(PROCESSED_PATH):
        raise HTTPException(status_code=400, detail="No processed data available. Call /upload first.")

    # Imported lazily so importing this module doesn't require matplotlib's
    # plotting backends to be usable at API import time.
    from src.models import classification, clustering, regression

    classification_comparison = classification.main()
    regression_comparison = regression.main()
    clustering_profile = clustering.main()

    load_state()

    return TrainResponse(
        classification_metrics=classification_comparison.reset_index().to_dict(orient="records"),
        regression_metrics=regression_comparison.reset_index().to_dict(orient="records"),
        clustering_profile=clustering_profile.reset_index().to_dict(orient="records"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(features: CustomerFeatures):
    feature_dict = features.model_dump()
    proba, prediction = _predict_churn(feature_dict)
    factors = _explain_churn_coefficients(feature_dict)
    return PredictionResponse(
        churn_probability=round(proba, 4), prediction=prediction,
        risk_factors=factors["risk_factors"], protective_factors=factors["protective_factors"],
    )


@app.get("/explain/{customer_id}", response_model=ExplainResponse)
def explain_customer(customer_id: str):
    row = _customer_row(customer_id)
    proba, prediction = _predict_churn(row)
    coefficient_based = _explain_churn_coefficients(row)

    shap_based = None
    shap_error = None
    try:
        shap_based = _explain_churn_shap(row)
    except Exception as e:  # noqa: BLE001 (SHAP is a best-effort comparison, never fatal)
        shap_error = str(e)

    return ExplainResponse(
        customer_id=customer_id,
        churn_probability=round(proba, 4),
        prediction=prediction,
        coefficient_based=ExplanationBreakdown(**coefficient_based),
        shap_based=ExplanationBreakdown(**shap_based) if shap_based else None,
        shap_error=shap_error,
    )


@app.post("/segment", response_model=SegmentResponse)
def segment_customer(request: SegmentRequest):
    if request.customer_id:
        row = _customer_row(request.customer_id)
    elif request.features:
        row = request.features.model_dump()
    else:
        raise HTTPException(status_code=400, detail="Provide either customer_id or features")

    cluster, segment_label = _predict_segment(row)
    return SegmentResponse(cluster=cluster, segment_label=segment_label)


@app.get("/model-metrics", response_model=ModelMetricsResponse)
def model_metrics():
    import json

    def _read(path):
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    return ModelMetricsResponse(
        classification=_read(CLASSIFICATION_METRICS_PATH),
        regression=_read(REGRESSION_METRICS_PATH),
        clustering=_read(CLUSTERING_METRICS_PATH),
    )


@app.get("/statistics", response_model=StatisticsResponse)
def statistics():
    customers = state.get("customers", pd.DataFrame())
    if customers.empty:
        raise HTTPException(status_code=404, detail="No processed data available. Call /upload first.")
    return StatisticsResponse(**compute_statistics(customers))


@app.get("/customers/{customer_id}", response_model=CustomerRecordResponse)
def get_customer(customer_id: str):
    row = _customer_row(customer_id)
    proba, prediction = _predict_churn(row)
    cluster, segment_label = _predict_segment(row)
    return CustomerRecordResponse(
        customer_id=customer_id,
        record=row,
        churn_probability=round(proba, 4),
        prediction=prediction,
        cluster=cluster,
        segment_label=segment_label,
    )


# --- Coefficient/importance-based vs. SHAP: comparing the two approaches ---
#
# /predict uses the coefficient-based method (_explain_churn_coefficients)
# by default: it's instant, has zero extra dependencies, and is exact for
# our current best model (Logistic Regression), because coefficient *
# scaled_value literally *is* that feature's contribution to the predicted
# log-odds — there's no approximation involved. /explain/{customer_id}
# additionally runs SHAP (_explain_churn_shap) so the two can be compared
# side by side for the same customer.
#
# For a linear model the two methods are near-identical by construction —
# shap.LinearExplainer computes contribution as coefficient * (value -
# background_mean), which collapses to the same coefficient * scaled_value
# used above whenever the background sample's mean is close to 0 (true
# here, since our features are already standard-scaled). Spot-checking three
# real customers: two came back with identical top-4/top-3 rankings; the
# third matched on 3 of 4 risk factors and all 3 protective factors, but
# disagreed on the 4th (weakest) risk factor — "High support-ticket rate"
# (coefficient) vs. "Female gender" (SHAP), two features whose raw
# contributions were close enough that the background-mean centering
# tipped the ranking. So: the two methods agree strongly, but not always
# exactly, and any disagreement shows up only at the margin (the weakest,
# least confident factor), never on the dominant ones.
#
# The real difference shows up on tree-based models (Decision Tree, Random
# Forest), which is where SHAP is actually doing more than the
# coefficient-based fallback:
# - The coefficient method's tree-based path multiplies global
#   feature_importances_ (how much a feature matters *on average across the
#   whole forest*) by a precomputed correlation-sign heuristic and this
#   customer's value. It's cheap and directionally reasonable, but it's an
#   approximation — the "direction" is a dataset-wide average, not
#   something derived from this specific customer's prediction path.
# - shap.TreeExplainer computes each feature's *actual* contribution to
#   this specific customer's output by walking the real decision paths of
#   every tree, so it correctly captures interaction effects (e.g. support
#   tickets mattering more for low-tenure customers than high-tenure ones)
#   that a single global importance score times a sign can't represent.
#
# Trade-off: SHAP is slower (has to walk tree structures or use a
# background sample) and adds a real dependency (shap, plus numba/llvmlite
# under the hood — installed fine here on Python 3.14, but that's a very
# recent, still-fragile combination worth watching in CI). The
# coefficient-based method has no such risk and is the right default for a
# linear model; SHAP earns its cost specifically for tree-based models,
# which is exactly where the coefficient method is weakest.
