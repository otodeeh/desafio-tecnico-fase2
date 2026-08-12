"""Contexto factual para que a LLM explique sem recalcular o diagnóstico."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class FeatureEvidence:
    feature: str
    value: float
    influence: float
    direction: str


@dataclass(frozen=True)
class PredictionContext:
    predicted_class: str
    malignant_probability: float
    threshold: float
    evidence: list[FeatureEvidence]
    model_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "evidence": [asdict(item) for item in self.evidence],
        }


def _logistic_evidence(model: Pipeline, row: pd.DataFrame, top_k: int) -> list[FeatureEvidence]:
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    scaled = scaler.transform(row)[0]
    contributions = scaled * classifier.coef_[0]
    ranking = np.argsort(np.abs(contributions))[::-1][:top_k]
    return [
        FeatureEvidence(
            feature=str(row.columns[index]),
            value=float(row.iloc[0, index]),
            influence=float(contributions[index]),
            direction="aumenta risco" if contributions[index] >= 0 else "reduz risco",
        )
        for index in ranking
    ]


def _forest_evidence(
    model: RandomForestClassifier,
    row: pd.DataFrame,
    top_k: int,
) -> list[FeatureEvidence]:
    ranking = np.argsort(model.feature_importances_)[::-1][:top_k]
    return [
        FeatureEvidence(
            feature=str(row.columns[index]),
            value=float(row.iloc[0, index]),
            influence=float(model.feature_importances_[index]),
            direction="importância global; direção não inferida",
        )
        for index in ranking
    ]


def prediction_context(
    model: Any,
    row: pd.DataFrame,
    *,
    model_name: str,
    threshold: float = 0.5,
    top_k: int = 5,
) -> PredictionContext:
    """Gera evidências numéricas; a função não produz aconselhamento clínico."""

    if len(row) != 1:
        raise ValueError("A explicação deve receber exatamente uma observação.")
    probability = float(model.predict_proba(row)[0, 1])
    predicted_class = "maligno" if probability >= threshold else "benigno"
    if isinstance(model, Pipeline):
        evidence = _logistic_evidence(model, row, top_k)
    elif isinstance(model, RandomForestClassifier):
        evidence = _forest_evidence(model, row, top_k)
    else:
        raise TypeError(f"Modelo sem explicador implementado: {type(model).__name__}.")
    return PredictionContext(
        predicted_class=predicted_class,
        malignant_probability=probability,
        threshold=threshold,
        evidence=evidence,
        model_name=model_name,
    )

