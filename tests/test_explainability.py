from tech_challenge_fase2.data import load_diagnostic_dataset, split_dataset
from tech_challenge_fase2.explainability import prediction_context
from tech_challenge_fase2.models import build_model


def test_prediction_context_contains_only_ranked_model_evidence():
    split = split_dataset(load_diagnostic_dataset())
    model = build_model("logistic_regression")
    model.fit(split.X_train, split.y_train)
    context = prediction_context(
        model,
        split.X_test.iloc[[0]],
        model_name="logistic_regression",
        top_k=5,
    )

    assert context.predicted_class in {"benigno", "maligno"}
    assert 0 <= context.malignant_probability <= 1
    assert len(context.evidence) == 5

