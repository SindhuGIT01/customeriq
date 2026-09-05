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

from src.evaluation.metrics import evaluate_classifier
from src.features.engineering import encode_and_scale, engineer_features

DATA_PATH = "data/processed/customers_clean.csv"
MODELS_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2
NON_FEATURE_COLUMNS = ["customer_id", "churn"]


def load_features(path: str = DATA_PATH):
    """Load cleaned data and turn it into a model-ready feature matrix."""
    df = pd.read_csv(path)
    df = engineer_features(df)
    transformed, _encoder, _scaler = encode_and_scale(df)
    X = transformed.drop(columns=NON_FEATURE_COLUMNS)
    y = transformed["churn"].astype(int)
    return X, y


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    }


def main():
    X, y = load_features()
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
    model_path = os.path.join(MODELS_DIR, "best_churn_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")

    return comparison


if __name__ == "__main__":
    main()

# --- Why recall matters more than raw accuracy here ------------------------
#
# Only ~6% of customers in this dataset churn, so a model that never predicts
# churn at all still scores ~94% accuracy while being completely useless for
# the business — Random Forest does exactly this above (94.2% accuracy, but
# 0% recall: it flagged zero of the 29 churners in the test set). The real
# cost asymmetry is: missing a customer who is about to churn (a false
# negative) means losing them with no chance to intervene, while flagging a
# loyal customer as at-risk (a false positive) just costs a wasted retention
# offer/email. Since a missed churner is far more expensive than a wasted
# outreach, recall on the churn class matters more than overall accuracy for
# this use case.
#
# --- Overfitting / bias-variance observations (actual run above) -----------
#
# - Decision Tree: 100% train accuracy vs. 88.0% test accuracy — a ~12-point
#   gap, the clearest sign of overfitting here. An unconstrained tree
#   memorized the training rows, including noise, then generalized poorly.
#   It still ends up the "best" model by F1 (0.091) simply because it's the
#   only one that catches more than one churner (3 of 29, recall 10.3%),
#   at the cost of 34 false positives.
# - Random Forest: also overfits train (99.9% vs. 94.2% test accuracy) but
#   in a different way — despite averaging many trees, it converges to
#   *always* predicting "no churn" on unseen data (0% recall). With a ~6%
#   minority class and no class weighting, majority voting across trees
#   washes out the minority-class splits entirely.
# - Logistic Regression: the smallest train/test gap (94.1% vs. 94.4%,
#   essentially identical) — low variance, but high bias toward the
#   majority class: it only catches 1 of 29 churners (recall 3.4%), since a
#   single linear boundary through 40+ features can't carve out the
#   minority region well.
#
# None of the three models were given class weighting to compensate for the
# ~6% churn rate, which is the main reason recall is low across the board.
# class_weight="balanced" or resampling (e.g. SMOTE) would be the natural
# next step to raise recall without changing the feature set.
