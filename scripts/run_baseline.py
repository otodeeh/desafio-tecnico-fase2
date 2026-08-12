"""Treina e registra os modelos originais antes da otimização genética."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tech_challenge_fase2.data import load_diagnostic_dataset, split_dataset
from tech_challenge_fase2.models import MODEL_NAMES, build_model, classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/baseline.json"),
    )
    args = parser.parse_args()

    dataset = load_diagnostic_dataset(args.data)
    split = split_dataset(dataset)
    results: dict[str, object] = {
        "dataset_source": split.source,
        "train_samples": len(split.X_train),
        "test_samples": len(split.X_test),
        "models": {},
    }
    for model_name in MODEL_NAMES:
        model = build_model(model_name)
        model.fit(split.X_train, split.y_train)
        results["models"][model_name] = classification_metrics(  # type: ignore[index]
            model, split.X_test, split.y_test
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

