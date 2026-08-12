from tech_challenge_fase2.explainability import FeatureEvidence, PredictionContext
from types import SimpleNamespace

from tech_challenge_fase2.llm import (
    ClinicalExplanation,
    DeterministicExplanationProvider,
    OpenAIExplanationProvider,
    build_user_prompt,
)


def test_offline_explanation_is_labeled_and_safe():
    context = PredictionContext(
        predicted_class="maligno",
        malignant_probability=0.91,
        threshold=0.5,
        evidence=[FeatureEvidence("radius_worst", 25.0, 1.2, "aumenta risco")],
        model_name="logistic_regression",
    )
    provider = DeterministicExplanationProvider()
    explanation = provider.explain(context)
    prompt = build_user_prompt(context)

    assert provider.name == "offline-template"
    assert "não substitui diagnóstico" in explanation.disclaimer
    assert "0.91" in prompt
    assert "radius_worst" in prompt


def test_openai_provider_uses_structured_responses_without_network():
    context = PredictionContext(
        predicted_class="benigno",
        malignant_probability=0.12,
        threshold=0.5,
        evidence=[FeatureEvidence("radius_mean", 8.0, -0.8, "reduz risco")],
        model_name="logistic_regression",
    )
    expected = ClinicalExplanation(
        summary="Resultado estatístico benigno.",
        supporting_evidence=["radius_mean=8.0"],
        limitations=["Dataset educacional."],
        recommended_action="Avaliação profissional.",
        disclaimer="Não substitui diagnóstico médico.",
    )
    captured = {}

    class FakeResponses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_parsed=expected)

    provider = OpenAIExplanationProvider(model="modelo-teste", api_key="chave-teste")
    provider.client = SimpleNamespace(responses=FakeResponses())

    assert provider.explain(context) == expected
    assert captured["model"] == "modelo-teste"
    assert captured["text_format"] is ClinicalExplanation
