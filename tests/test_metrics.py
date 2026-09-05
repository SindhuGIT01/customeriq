from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import evaluate_classifier


def test_evaluate_classifier_returns_expected_keys_and_ranges():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = LogisticRegression().fit(X, y)

    result = evaluate_classifier(model, X, y)

    assert set(result.keys()) == {"accuracy", "precision", "recall", "f1", "confusion_matrix"}
    for key in ["accuracy", "precision", "recall", "f1"]:
        assert 0.0 <= result[key] <= 1.0
    assert result["confusion_matrix"].shape == (2, 2)
