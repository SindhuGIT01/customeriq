# CustomerIQ

![CI](https://github.com/SindhuGIT01/customeriq/actions/workflows/ci.yml/badge.svg)

CustomerIQ — ML-Powered Customer Analytics & Churn Intelligence Platform

## Overview

CustomerIQ is an end-to-end machine learning platform for analyzing customer
behavior: it ingests raw customer records, cleans and engineers features,
trains models to predict churn and customer lifetime value (CLV), segments
customers into behavioral clusters, explains individual predictions, and
serves all of it through a FastAPI service.

Since a public dataset with the exact schema this project needs doesn't
exist, the pipeline runs on a generated synthetic telecom/SaaS dataset with
realistic, internally consistent relationships (e.g. churn correlates with
low usage and high support-ticket volume; total spend tracks tenure ×
monthly rate) — see `src/data/generate_synthetic_data.py`.

## Architecture

```
 ┌─────────────────────┐
 │  Raw customer CSV     │   data/raw/customers_raw.csv
 └──────────┬───────────┘
            │ extract()
            ▼
 ┌─────────────────────┐
 │  ETL pipeline          │   src/data/cleaning.py
 │  validate → transform  │   src/data/preprocessing.py
 └──────────┬───────────┘
            ▼
 ┌─────────────────────┐
 │  Cleaned dataset       │   data/processed/customers_clean.csv
 └──────────┬───────────┘
            │ engineer_features() + encode_and_scale()
            ▼
 ┌─────────────────────┐
 │  Feature engineering   │   src/features/engineering.py
 └──────────┬───────────┘
            │
   ┌────────┼─────────────────┐
   ▼        ▼                 ▼
┌────────┐┌────────┐    ┌────────────┐
│Classify││Regress │    │ Clustering  │   src/models/*.py
│(churn) ││ (CLV)  │    │ (segments)  │
└───┬────┘└───┬────┘    └─────┬──────┘
    │         │               │
    └────┬────┴────────┬──────┘
         ▼              ▼
  models/*.joblib  models/*.json      (model bundles + metrics)
         │              │
         └──────┬───────┘
                ▼
   ┌─────────────────────────────┐
   │          FastAPI               │   src/api/main.py
   │ /upload  /train  /predict       │
   │ /segment  /explain/{id}          │
   │ /model-metrics  /statistics      │
   │ /customers/{id}                  │
   └─────────────────────────────┘
```

## Project Structure

```
data/raw/             Raw, unprocessed data (gitignored, regenerate via script)
data/processed/       Cleaned and segmented data (gitignored, regenerate via script)
models/               Trained model bundles + metrics JSON (gitignored, regenerate via script)
reports/              Generated plots (EDA, elbow curve, cluster scatter, CLV fit)
src/data/             Data generation, cleaning, and the ETL pipeline
src/features/         Feature engineering (ratios, buckets, one-hot + scaling)
src/models/           Classification, regression, and clustering training
src/evaluation/       Metrics helpers, statistics dashboard, A/B testing
src/api/              FastAPI service
tests/                Unit and API tests
notebooks/            Exploratory data analysis
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows; use `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

Generate the data and train all models (each step's output feeds the next):

```bash
python -m src.data.generate_synthetic_data   # -> data/raw/customers_raw.csv
python -m src.data.preprocessing             # -> data/processed/customers_clean.csv
python -m src.models.classification          # -> models/best_churn_model.joblib
python -m src.models.regression              # -> models/best_clv_model.joblib
python -m src.models.clustering              # -> models/clustering_model.joblib, data/processed/customers_segmented.csv
```

Run the API:

```bash
uvicorn src.api.main:app --reload
```

Run the tests:

```bash
python -m pytest
ruff check .
```

## API Reference

All examples below are real request/response pairs captured from a running
instance of this service.

### `POST /upload` — ingest a raw CSV, run it through cleaning + validation

```bash
curl -X POST http://127.0.0.1:8000/upload -F "file=@data/raw/customers_raw.csv"
```
```json
{
  "rows_ingested": 2537,
  "rows_after_cleaning": 2490,
  "message": "Data cleaned, feature-engineering validated, and stored as the current processed dataset."
}
```

### `POST /train` — retrain all three models on the current processed data

```bash
curl -X POST http://127.0.0.1:8000/train
```
```json
{
  "classification_metrics": [
    {"model": "Logistic Regression", "test_accuracy": 0.671, "precision": 0.086, "recall": 0.483, "f1": 0.146},
    {"model": "Decision Tree", "test_accuracy": 0.894, "precision": 0.038, "recall": 0.034, "f1": 0.036},
    {"model": "Random Forest", "test_accuracy": 0.940, "precision": 0.0, "recall": 0.0, "f1": 0.0}
  ],
  "regression_metrics": [
    {"model": "Linear Regression", "mae": 824.31, "rmse": 1431.41, "r2": 0.634},
    {"model": "Random Forest Regressor", "mae": 374.42, "rmse": 1157.50, "r2": 0.761}
  ],
  "clustering_profile": [
    {"cluster": 0, "segment_label": "High-Value Loyal", "size": 499},
    {"cluster": 1, "segment_label": "Low-Value", "size": 1439},
    {"cluster": 2, "segment_label": "New Customer", "size": 176},
    {"cluster": 3, "segment_label": "High-Value At-Risk", "size": 376}
  ]
}
```

### `POST /predict` — churn probability + explanation for a customer's features

Any field may be omitted; missing values are filled from the training set's
per-column median/mode.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure_months": 8, "monthly_spend": 1499, "login_frequency": 3, "support_tickets": 5}'
```
```json
{
  "churn_probability": 0.9998,
  "prediction": "High Risk",
  "risk_factors": ["High monthly spend", "Many support tickets", "Short tenure", "Low total spend"],
  "protective_factors": ["Low customer lifetime value", "Male gender", "Los Angeles location"]
}
```

### `POST /segment` — which customer segment a customer belongs to

```bash
curl -X POST http://127.0.0.1:8000/segment \
  -H "Content-Type: application/json" -d '{"customer_id": "CUST101806"}'
```
```json
{"cluster": 3, "segment_label": "High-Value At-Risk"}
```

### `GET /model-metrics` — latest saved metrics for all three models

```bash
curl http://127.0.0.1:8000/model-metrics
```
```json
{
  "classification": {"best_model": "Logistic Regression", "metrics": ["..."]},
  "regression": {"best_model": "Random Forest Regressor", "metrics": ["..."]},
  "clustering": {"chosen_k": 4, "profile": ["..."]}
}
```

### `GET /statistics` — dataset statistics dashboard

```bash
curl http://127.0.0.1:8000/statistics
```
```json
{
  "customer_count": 2490,
  "churn_rate": 0.059,
  "summary_statistics": {"monthly_spend": {"mean": 54.13, "median": 43.71, "std": 38.07}, "...": "..."},
  "missing_value_counts": {},
  "duplicate_customer_id_count": 0,
  "correlation_matrix": {"...": "..."}
}
```

### `GET /customers/{id}` — stored record + churn prediction + segment

```bash
curl http://127.0.0.1:8000/customers/CUST101806
```
```json
{
  "customer_id": "CUST101806",
  "record": {"age": 48, "gender": "Female", "tenure_months": 34, "monthly_spend": 132.19, "subscription_type": "Enterprise", "...": "..."},
  "churn_probability": 0.2434,
  "prediction": "Low Risk",
  "cluster": 3,
  "segment_label": "High-Value At-Risk"
}
```

### `GET /explain/{id}` — coefficient-based vs. SHAP explanation, side by side

```bash
curl http://127.0.0.1:8000/explain/CUST101806
```
```json
{
  "customer_id": "CUST101806",
  "churn_probability": 0.2434,
  "prediction": "Low Risk",
  "coefficient_based": {
    "risk_factors": ["High customer lifetime value", "High monthly spend", "High average monthly spend", "PayPal payment method"],
    "protective_factors": ["High total spend", "Long tenure", "High product usage"]
  },
  "shap_based": {
    "risk_factors": ["High customer lifetime value", "High monthly spend", "High average monthly spend", "PayPal payment method"],
    "protective_factors": ["High total spend", "Long tenure", "High product usage"]
  },
  "shap_error": null
}
```

## Results

| Task | Best model | Key metric |
|---|---|---|
| Churn classification | Logistic Regression (`class_weight="balanced"`) | **F1 = 0.146** (precision 0.086, recall 0.483) |
| CLV regression | Random Forest Regressor | **R² = 0.761** (MAE $374, RMSE $1,158) |
| Customer segmentation | K-Means | **K = 4** (chosen via elbow method) |

**Churn**: the ~6% churn rate makes this a hard imbalance problem — a
model that never predicts churn scores ~94% accuracy while being useless.
Logistic Regression with balanced class weights was chosen for the best F1,
trading a lot of precision for much higher recall (catches ~48% of actual
churners vs. ~3% unweighted). See `src/models/classification.py`'s trailing
comment block for the full bias/variance write-up.

**CLV**: Random Forest Regressor explains 76% of the variance in customer
lifetime value (vs. 63% for Linear Regression) and its typical prediction
error is less than half — $374 vs. $824 per customer — because it captures
non-linear interactions a straight-line model can't (e.g. support tickets
hurting CLV more for low-tenure customers).

**Segmentation**: the elbow method chose K=4, producing four distinct,
business-actionable segments: **High-Value Loyal** (499 customers, 4.5yr
avg. tenure, 0.6% churn), **Low-Value** (1,439 customers, the majority,
below-average CLV), **New Customer** (176 customers, ~1 month tenure), and
**High-Value At-Risk** (376 customers — the highest spenders, but churning
at 9%, the same elevated rate as brand-new customers).
