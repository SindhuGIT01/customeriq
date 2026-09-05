# CustomerIQ — 3-Minute Demo Script

Walkthrough for a live or recorded demo of the API. Assumes the service is
already running (`uvicorn src.api.main:app --reload`) and the models are
already trained (see the Setup section of [README.md](README.md)). Every
command and response below is real output from a running instance — copy
them verbatim if recording, or swap in your own customer_id.

---

### 0:00–0:20 — Intro

> "This is CustomerIQ, an end-to-end churn intelligence platform. Raw
> customer data comes in as a CSV, gets cleaned and feature-engineered, and
> feeds three models — churn classification, CLV regression, and customer
> segmentation — all served through one FastAPI backend. Let's walk through
> the actual API."

---

### 0:20–0:50 — Upload data

> "First, I'll upload a batch of customer records."

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

> "That ran real validation and cleaning — it ingested 2537 rows and kept
> 2490 after removing duplicates and dropping the handful of rows with no
> churn label. This is now the live dataset the rest of the API serves
> from."

---

### 0:50–1:30 — Predict churn for a single customer

> "Now let's score a customer who's been with us 8 months, has an
> unusually high bill, and has opened 5 support tickets while barely
> logging in."

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

> "99.98% churn probability, flagged High Risk — and it's not a black box.
> The response itself names the top risk factors driving that score, right
> next to the prediction."

---

### 1:30–2:00 — Segment an existing customer

> "Every customer also belongs to a behavioral segment from K-Means
> clustering. Let's check a real one."

```bash
curl -X POST http://127.0.0.1:8000/segment \
  -H "Content-Type: application/json" -d '{"customer_id": "CUST101806"}'
```

```json
{"cluster": 3, "segment_label": "High-Value At-Risk"}
```

> "This customer lands in 'High-Value At-Risk' — our highest-spending
> segment, which churns at 9%, the same elevated rate as brand-new
> customers. That's a genuinely actionable finding: our best accounts need
> just as much retention attention as day-one signups."

---

### 2:00–2:40 — Explain the prediction two ways

> "For that same customer, here's the full explanation — comparing two
> independent methods side by side."

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

> "It runs the model's own coefficients *and* a SHAP explanation, and shows
> both. Here they agree almost exactly — long tenure and high product
> usage are protecting this customer despite their high spend nudging risk
> up. Two independent methods landing on the same answer is a real sanity
> check that the explanation is trustworthy, not an artifact of one
> method's assumptions."

---

### 2:40–3:00 — Wrap-up

> "That's the full loop: raw data in, cleaned and engineered, three models
> trained and served, every prediction explained in plain English — one
> API, with CI running the test suite and linter on every push."

---

## Optional extras (if there's time)

- `GET /statistics` — the live dataset's summary stats, missingness, and
  correlation matrix.
- `GET /model-metrics` — the last-trained metrics for all three models,
  without needing to retrain.
- `POST /train` — retrain all three models on whatever's currently
  uploaded, and reload them into the running service.
