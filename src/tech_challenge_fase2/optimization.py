"""Avaliação por validação cruzada e execução dos experimentos genéticos."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from .ga import GAConfig, GAResult, GeneticAlgorithm
from .models import build_model, decode_parameters, parameter_space


@dataclass(frozen=True)
class ExperimentResult:
    model_name: str
    config: dict[str, Any]
    best_genes: dict[str, Any]
    best_parameters: dict[str, Any]
    best_fitness: float
    cv_metrics: dict[str, float]
    history: list[dict[str, float | int]]
    evaluations: int


def experiment_configs() -> tuple[GAConfig, ...]:
    """As três configurações obrigatórias de população e operadores."""

    return (
        GAConfig(
            name="compacta",
            population_size=10,
            generations=6,
            crossover_rate=0.80,
            mutation_rate=0.10,
            tournament_size=3,
            elitism=2,
            random_state=101,
        ),
        GAConfig(
            name="balanceada",
            population_size=14,
            generations=8,
            crossover_rate=0.85,
            mutation_rate=0.20,
            tournament_size=3,
            elitism=2,
            random_state=202,
        ),
        GAConfig(
            name="exploratoria",
            population_size=18,
            generations=10,
            crossover_rate=0.70,
            mutation_rate=0.30,
            tournament_size=4,
            elitism=2,
            random_state=303,
        ),
    )


def make_fitness_function(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    cv_splits: int = 5,
    random_state: int = 42,
):
    """Combina recall (65%) e F1 (35%), priorizando falsos negativos."""

    cross_validator = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )

    def evaluate(genes: dict[str, Any]) -> tuple[float, dict[str, float]]:
        parameters = decode_parameters(model_name, genes)
        model = build_model(model_name, parameters, random_state=random_state)
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cross_validator,
            scoring={
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
                "roc_auc": "roc_auc",
            },
            n_jobs=1,
            error_score="raise",
        )
        metrics = {
            name: float(np.mean(scores[f"test_{name}"]))
            for name in ("accuracy", "precision", "recall", "f1", "roc_auc")
        }
        fitness = 0.65 * metrics["recall"] + 0.35 * metrics["f1"]
        metrics["fitness"] = float(fitness)
        return float(fitness), metrics

    return evaluate


def run_experiment(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: GAConfig,
    *,
    cv_splits: int = 5,
) -> ExperimentResult:
    fitness_function = make_fitness_function(
        model_name,
        X_train,
        y_train,
        cv_splits=cv_splits,
        random_state=config.random_state,
    )
    ga = GeneticAlgorithm(parameter_space(model_name), fitness_function, config)
    result: GAResult = ga.run()
    return ExperimentResult(
        model_name=model_name,
        config=asdict(config),
        best_genes=result.best_genes,
        best_parameters=decode_parameters(model_name, result.best_genes),
        best_fitness=result.best_fitness,
        cv_metrics=result.best_details,
        history=result.history,
        evaluations=result.evaluations,
    )

