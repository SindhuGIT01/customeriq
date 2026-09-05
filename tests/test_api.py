"""API tests. Model/data artifacts are gitignored and won't exist in a
fresh checkout (e.g. CI), so these build a small self-contained set of
bundles in a temp dir and monkeypatch the API module to use them, rather
than depending on the real trained models."""

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import src.api.main as api_main
from src.data.cleaning import clean_data
from src.data.generate_synthetic_data import generate
from src.features.engineering import encode_and_scale, engineer_features
from src.models.clustering import CLUSTER_FEATURES


def _build_test_bundles(tmp_path, n_rows=80, seed=7):
    raw_df = generate(n_rows=n_rows, seed=seed)
    clean_df = clean_data(raw_df)

    processed_path = tmp_path / "customers_clean.csv"
    clean_df.to_csv(processed_path, index=False)

    engineered_df = engineer_features(clean_df)
    transformed, encoder, scaler = encode_and_scale(engineered_df)
    X = transformed.drop(columns=["customer_id", "churn"])
    y = transformed["churn"].astype(int)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X, y)
    feature_directions = {col: (1.0 if X[col].corr(y) >= 0 else -1.0) for col in X.columns}

    churn_model_path = tmp_path / "best_churn_model.joblib"
    joblib.dump(
        {"model": model, "encoder": encoder, "scaler": scaler, "feature_directions": feature_directions},
        churn_model_path,
    )

    cluster_scaler = StandardScaler()
    X_cluster = cluster_scaler.fit_transform(engineered_df[CLUSTER_FEATURES])
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    cluster_model_path = tmp_path / "clustering_model.joblib"
    joblib.dump(
        {"kmeans": kmeans, "scaler": cluster_scaler, "segment_labels": {0: "Segment A", 1: "Segment B"}},
        cluster_model_path,
    )

    return processed_path, churn_model_path, cluster_model_path


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    processed_path, churn_model_path, cluster_model_path = _build_test_bundles(tmp_path)

    monkeypatch.setattr(api_main, "PROCESSED_PATH", str(processed_path))
    monkeypatch.setattr(api_main, "CHURN_MODEL_PATH", str(churn_model_path))
    monkeypatch.setattr(api_main, "CLUSTER_MODEL_PATH", str(cluster_model_path))

    with TestClient(api_main.app) as client:
        yield client


def test_predict_endpoint_returns_valid_response(api_client):
    response = api_client.post("/predict", json={
        "tenure_months": 8, "monthly_spend": 1499, "login_frequency": 3, "support_tickets": 5,
    })

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["prediction"] in ("High Risk", "Low Risk")
    assert isinstance(body["risk_factors"], list)
    assert isinstance(body["protective_factors"], list)


def test_predict_endpoint_handles_empty_payload(api_client):
    """All fields are optional; an empty body should fall back to defaults."""
    response = api_client.post("/predict", json={})
    assert response.status_code == 200
    assert 0.0 <= response.json()["churn_probability"] <= 1.0


def test_predict_returns_503_when_model_not_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "PROCESSED_PATH", str(tmp_path / "missing.csv"))
    monkeypatch.setattr(api_main, "CHURN_MODEL_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setattr(api_main, "CLUSTER_MODEL_PATH", str(tmp_path / "missing_cluster.joblib"))

    with TestClient(api_main.app) as client:
        response = client.post("/predict", json={"tenure_months": 5})

    assert response.status_code == 503


def test_statistics_endpoint_returns_valid_response(api_client):
    response = api_client.get("/statistics")

    assert response.status_code == 200
    body = response.json()
    assert body["customer_count"] > 0
    assert 0.0 <= body["churn_rate"] <= 1.0
    assert "monthly_spend" in body["summary_statistics"]
    assert "mean" in body["summary_statistics"]["monthly_spend"]


def test_statistics_returns_404_when_no_data(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "PROCESSED_PATH", str(tmp_path / "missing.csv"))
    monkeypatch.setattr(api_main, "CHURN_MODEL_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setattr(api_main, "CLUSTER_MODEL_PATH", str(tmp_path / "missing_cluster.joblib"))

    with TestClient(api_main.app) as client:
        response = client.get("/statistics")

    assert response.status_code == 404
