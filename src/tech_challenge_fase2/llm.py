"""Integração segura com LLM para explicar resultados já calculados."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .explainability import PredictionContext


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


SYSTEM_PROMPT = """Você é um assistente de comunicação clínica para uso educacional.
Explique somente os fatos presentes no JSON recebido. Não altere o diagnóstico,
não invente sintomas, histórico, causas, tratamentos ou valores. Diferencie
claramente a saída estatística do modelo de um diagnóstico médico. Seja direto,
use português do Brasil e recomende avaliação por profissional habilitado.
"""


class ClinicalExplanation(BaseModel):
    summary: str = Field(description="Resumo factual do resultado do modelo.")
    supporting_evidence: list[str] = Field(
        description="Evidências numéricas presentes no contexto fornecido."
    )
    limitations: list[str] = Field(
        description="Limitações do modelo e da interpretação."
    )
    recommended_action: str = Field(
        description="Próximo passo seguro, sem prescrever tratamento."
    )
    disclaimer: str = Field(
        description="Aviso de que a saída não substitui diagnóstico médico."
    )


class ExplanationProvider(Protocol):
    name: str

    def explain(self, context: PredictionContext) -> ClinicalExplanation: ...


def build_user_prompt(context: PredictionContext) -> str:
    return (
        "Explique o resultado abaixo respeitando estritamente o schema solicitado.\n"
        + json.dumps(context.to_dict(), ensure_ascii=False, indent=2)
    )


class OpenAIExplanationProvider:
    """Provider real usando Responses API e saída estruturada por Pydantic."""

    name = "openai"

    def __init__(self, *, model: str | None = None, api_key: str | None = None) -> None:
        from openai import OpenAI

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def explain(self, context: PredictionContext) -> ClinicalExplanation:
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(context)},
            ],
            text_format=ClinicalExplanation,
        )
        if response.output_parsed is None:
            raise RuntimeError("A LLM não retornou uma explicação estruturada.")
        return response.output_parsed


class DeterministicExplanationProvider:
    """Fallback offline para testes; não se apresenta como resposta de LLM."""

    name = "offline-template"

    def explain(self, context: PredictionContext) -> ClinicalExplanation:
        percentage = context.malignant_probability * 100
        evidence = [
            f"{item.feature}={item.value:.4g} ({item.direction}, influência={item.influence:.4g})"
            for item in context.evidence
        ]
        return ClinicalExplanation(
            summary=(
                f"O modelo {context.model_name} classificou o caso como "
                f"{context.predicted_class}, com probabilidade estimada de "
                f"malignidade de {percentage:.1f}%."
            ),
            supporting_evidence=evidence,
            limitations=[
                "A estimativa depende do conjunto de treinamento e das variáveis disponíveis.",
                "A importância das variáveis não demonstra causalidade.",
            ],
            recommended_action="Encaminhar o resultado para avaliação de profissional habilitado.",
            disclaimer="Uso educacional; esta saída não substitui diagnóstico médico.",
        )


def provider_from_environment() -> ExplanationProvider:
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIExplanationProvider()
    return DeterministicExplanationProvider()
