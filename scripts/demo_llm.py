"""Demonstra a explicação por LLM ou pelo fallback offline identificado."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from tech_challenge_fase2.data import load_diagnostic_dataset
from tech_challenge_fase2.explainability import prediction_context
from tech_challenge_fase2.llm import provider_from_environment


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/models/logistic_regression_best.joblib"),
    )
    parser.add_argument("--row", type=int, default=0)
    args = parser.parse_args()

    artifact = joblib.load(args.artifact)
    dataset = load_diagnostic_dataset()
    row = dataset.features.iloc[[args.row]][artifact["feature_names"]]
    context = prediction_context(
        artifact["model"],
        row,
        model_name=artifact["model_name"],
    )
    provider = provider_from_environment()
    explanation = provider.explain(context)
    print(
        json.dumps(
            {"provider": provider.name, "context": context.to_dict(), "explanation": explanation.model_dump()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
