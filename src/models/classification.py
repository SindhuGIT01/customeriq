"""Train and compare churn classification models.

Loads the cleaned dataset, engineers features, trains Logistic Regression,
Decision Tree, and Random Forest classifiers, evaluates each, and saves the
best-performing one (by F1 score) to models/.
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.evaluation.metrics import evaluate_classifier, save_metrics_json
from src.features.engineering import encode_and_scale, engineer_features

DATA_PATH = "data/processed/customers_clean.csv"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "best_churn_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "classification_metrics.json")
RANDOM_STATE = 42
TEST_SIZE = 0.2
NON_FEATURE_COLUMNS = ["customer_id", "churn"]


def load_features(path: str = DATA_PATH):
    """Load cleaned data and turn it into a model-ready feature matrix.

    Returns the fitted encoder/scaler too so callers (e.g. the API, which
    needs to transform a single new customer the same way at prediction
    time) can persist and reuse them instead of refitting.
    """
    df = pd.read_csv(path)
    df = engineer_features(df)
    transformed, encoder, scaler = encode_and_scale(df)
    X = transformed.drop(columns=NON_FEATURE_COLUMNS)
    y = transformed["churn"].astype(int)
    return X, y, encoder, scaler


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
    }


def main():
    X, y, encoder, scaler = load_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    fitted_models = {}
    rows = []

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

        train_metrics = evaluate_classifier(model, X_train, y_train)
        test_metrics = evaluate_classifier(model, X_test, y_test)

        print(f"\n{name}")
        print(f"  Confusion matrix (test):\n{test_metrics['confusion_matrix']}")

        rows.append({
            "model": name,
            "train_accuracy": train_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
        })

    comparison = pd.DataFrame(rows).set_index("model")
    print("\nModel comparison (test set, except train_accuracy):")
    print(comparison.to_string())

    best_model_name = comparison["f1"].idxmax()
    best_model = fitted_models[best_model_name]
    print(f"\nBest model by F1: {best_model_name}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    # Bundle the model with the encoder/scaler it was trained against, so a
    # caller (e.g. the prediction API) can transform new raw input the same
    # way without needing to refit anything.
    joblib.dump({"model": best_model, "encoder": encoder, "scaler": scaler}, MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")

    save_metrics_json(
        {"best_model": best_model_name, "metrics": comparison.reset_index().to_dict(orient="records")},
        METRICS_PATH,
    )
    print(f"Saved metrics to {METRICS_PATH}")

    return comparison


if __name__ == "__main__":
    main()

# --- Why recall matters more than raw accuracy here ------------------------
#
# Only ~6% of customers in this dataset churn, so a model that never predicts
# churn at all still scores ~94% accuracy while being completely useless for
# the business — Random Forest does exactly this above even with balanced
# class weights (93.98% accuracy, but 0% recall: it flagged zero of the 29
# churners in the test set). The real cost asymmetry is: missing a customer
# who is about to churn (a false negative) means losing them with no chance
# to intervene, while flagging a loyal customer as at-risk (a false positive)
# just costs a wasted retention offer/email. Since a missed churner is far
# more expensive than a wasted outreach, recall on the churn class matters
# more than overall accuracy for this use case.
#
# --- Effect of class_weight="balanced" (before -> after) -------------------
#
# Adding class_weight="balanced" to all three models had a very uneven
# effect — it is not a fix that helps every model equally:
# - Logistic Regression: recall jumped from 3.4% -> 48.3% (1 of 29 churners
#   caught -> 14 of 29), because reweighting directly changes where its
#   single linear boundary sits. The cost is a large drop in accuracy
#   (94.4% -> 67.1%) and precision (100% -> 8.6%, 149 false positives) — it
#   now over-flags heavily. F1 still improved (0.067 -> 0.146), making it
#   the new best model by that metric.
# - Decision Tree: barely changed and, on this split, got *worse* on the
#   minority class (recall 10.3% -> 3.4%, F1 0.091 -> 0.036). Reweighting
#   shifts split criteria but an unconstrained tree still ends up carving
#   out whichever noisy leaf boundaries best fit the training data, which
#   doesn't reliably translate into better minority-class splits on unseen
#   data.
# - Random Forest: no change at all (recall stayed at 0%). Averaging ~100
#   trees over a ~6% minority class continues to wash out minority
#   predictions even when each tree is fit on reweighted classes — majority
#   voting still favors "no churn" almost everywhere in feature space.
#
# Takeaway: class_weight="balanced" is not a universal fix — it moved the
# needle substantially for the linear model but did nothing (or slightly
# hurt) for the tree-based models here. Resampling (e.g. SMOTE), threshold
# tuning, or constraining tree depth would be more promising next steps for
# the tree-based models specifically.
#
# --- Overfitting / bias-variance observations (actual run above) -----------
#
# - Decision Tree: 100% train accuracy vs. 89.4% test accuracy — still the
#   clearest overfit here (unconstrained tree memorizes training rows,
#   including noise, then generalizes poorly), unchanged by class weighting.
# - Random Forest: also fully overfits train (100% vs. 94.0% test accuracy),
#   same pattern as before class weighting — ensembling reduces the training
#   memorization symptom somewhat but does not fix the recall collapse.
# - Logistic Regression: still the smallest train/test gap (66.8% vs. 67.1%,
#   virtually identical) — low variance either way. What changed is *where*
#   its bias points: unweighted it was biased toward the majority class
#   (barely predicting churn at all); balanced it's now biased toward
#   over-predicting churn instead. Same low-variance model, very different
#   operating point.
