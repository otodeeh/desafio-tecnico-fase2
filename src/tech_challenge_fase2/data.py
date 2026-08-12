"""Carregamento reproduzível do projeto de diagnóstico criado na Fase 1."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE1_DATA = PROJECT_ROOT.parent / "desafio-tecnico-fase1" / "data.csv"


@dataclass(frozen=True)
class DiagnosticDataset:
    """Features, alvo e rastreabilidade da fonte usada."""

    features: pd.DataFrame
    target: pd.Series
    source: str


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    source: str


def _candidate_paths(explicit_path: str | Path | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    if env_path := os.getenv("PHASE1_DATA_PATH"):
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            PROJECT_ROOT / "data" / "raw" / "data.csv",
            DEFAULT_PHASE1_DATA,
        ]
    )
    return [path.resolve() for path in candidates]


def load_diagnostic_dataset(path: str | Path | None = None) -> DiagnosticDataset:
    """Carrega o CSV da Fase 1; usa o dataset equivalente do sklearn como fallback."""

    for candidate in _candidate_paths(path):
        if not candidate.is_file():
            continue
        frame = pd.read_csv(candidate)
        unnamed = [column for column in frame.columns if column.startswith("Unnamed")]
        frame = frame.drop(columns=unnamed, errors="ignore")
        if "diagnosis" not in frame:
            raise ValueError(f"A coluna 'diagnosis' não existe em {candidate}.")
        target = frame.pop("diagnosis").map({"M": 1, "B": 0})
        if target.isna().any():
            raise ValueError("Foram encontrados diagnósticos diferentes de M e B.")
        features = frame.drop(columns=["id"], errors="ignore").astype(float)
        return DiagnosticDataset(
            features=features,
            target=target.astype(int).rename("malignant"),
            source=str(candidate),
        )

    sklearn_data = load_breast_cancer(as_frame=True)
    features = sklearn_data.data.copy()
    features.columns = [
        column.lower().replace(" ", "_").replace("error", "se")
        for column in features.columns
    ]
    # No sklearn, 0 representa maligno e 1 benigno. O projeto usa 1 para maligno.
    target = (sklearn_data.target == 0).astype(int).rename("malignant")
    return DiagnosticDataset(
        features=features.astype(float),
        target=target,
        source="sklearn.datasets.load_breast_cancer",
    )


def split_dataset(
    dataset: DiagnosticDataset,
    *,
    test_size: float = 0.12,
    random_state: int = 6546,
) -> DatasetSplit:
    """Replica a divisão estratificada principal registrada na Fase 1."""

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.features,
        dataset.target,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.target,
    )
    return DatasetSplit(X_train, X_test, y_train, y_test, dataset.source)

