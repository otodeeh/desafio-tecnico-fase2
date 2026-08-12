"""Modelos originais, decodificação de genes e métricas comparáveis."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .ga import GeneSpec


MODEL_NAMES = ("logistic_regression", "random_forest")


def parameter_space(model_name: str) -> dict[str, GeneSpec]:
    """Espaço de busca explícito usado como codificação genética."""

    if model_name == "logistic_regression":
        return {
            "log10_C": GeneSpec(kind="float", low=-4.0, high=4.0),
            "class_weight": GeneSpec(kind="choice", choices=(None, "balanced")),
            "solver": GeneSpec(kind="choice", choices=("lbfgs", "liblinear")),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": GeneSpec(kind="int", low=50, high=300),
            "max_depth": GeneSpec(kind="choice", choices=(None, 3, 5, 8, 12, 18)),
            "min_samples_split": GeneSpec(kind="int", low=2, high=12),
            "min_samples_leaf": GeneSpec(kind="int", low=1, high=6),
            "max_features": GeneSpec(kind="choice", choices=("sqrt", "log2", 0.5, 0.8)),
            "class_weight": GeneSpec(kind="choice", choices=(None, "balanced")),
        }
    raise ValueError(f"Modelo desconhecido: {model_name!r}.")


def decode_parameters(model_name: str, genes: dict[str, Any]) -> dict[str, Any]:
    """Converte o genótipo em hiperparâmetros aceitos pelo estimador."""

    decoded = dict(genes)
    if model_name == "logistic_regression":
        decoded["C"] = 10.0 ** float(decoded.pop("log10_C"))
    return decoded


def build_model(
    model_name: str,
    parameters: dict[str, Any] | None = None,
    *,
    random_state: int = 42,
) -> BaseEstimator:
    """Cria modelos reproduzíveis equivalentes aos usados na Fase 1."""

    parameters = dict(parameters or {})
    if model_name == "logistic_regression":
        defaults: dict[str, Any] = {
            "C": 1.0,
            "class_weight": None,
            "solver": "lbfgs",
        }
        defaults.update(parameters)
        classifier = LogisticRegression(
            **defaults,
            max_iter=3000,
            random_state=random_state,
        )
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )
    if model_name == "random_forest":
        defaults = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": None,
        }
        defaults.update(parameters)
        return RandomForestClassifier(
            **defaults,
            random_state=random_state,
            n_jobs=1,
        )
    raise ValueError(f"Modelo desconhecido: {model_name!r}.")


def classification_metrics(model: BaseEstimator, X: Any, y: Any) -> dict[str, Any]:
    """Calcula as métricas obrigatórias e mantém a matriz de confusão serializável."""

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    matrix = confusion_matrix(y, predictions)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "confusion_matrix": np.asarray(matrix, dtype=int).tolist(),
    }

