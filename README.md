# CustomerIQ

CustomerIQ — ML-Powered Customer Analytics & Churn Intelligence Platform

## Overview

CustomerIQ is a machine learning platform for analyzing customer behavior and
predicting churn, built to help businesses understand and retain their
customers.

## Project Structure

```
data/raw/            Raw, unprocessed data
data/processed/       Cleaned and feature-engineered data
src/data/             Data loading and preprocessing
src/features/         Feature engineering
src/models/           Model training and inference
src/evaluation/       Model evaluation and metrics
src/api/              API for serving predictions
tests/                Unit and integration tests
notebooks/            Exploratory analysis notebooks
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Testing

```bash
python -m pytest
```
