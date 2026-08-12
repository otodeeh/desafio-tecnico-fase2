from tech_challenge_fase2.data import load_diagnostic_dataset, split_dataset
from tech_challenge_fase2.models import build_model, classification_metrics


def test_logistic_baseline_reaches_expected_quality():
    split = split_dataset(load_diagnostic_dataset())
    model = build_model("logistic_regression")
    model.fit(split.X_train, split.y_train)
    metrics = classification_metrics(model, split.X_test, split.y_test)

    assert metrics["accuracy"] >= 0.90
    assert metrics["recall"] >= 0.90
    assert len(metrics["confusion_matrix"]) == 2

