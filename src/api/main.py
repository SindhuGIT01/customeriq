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


def _predict_churn(feature_dict: dict) -> tuple:
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

    proba = float(model.predict_proba(X)[0][1])
    prediction = "High Risk" if proba >= 0.5 else "Low Risk"
    return proba, prediction


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
    proba, prediction = _predict_churn(features.model_dump())
    return PredictionResponse(churn_probability=round(proba, 4), prediction=prediction)


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
