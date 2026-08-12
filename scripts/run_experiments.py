"""Executa três configurações de GA e compara baseline versus otimizado."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from tech_challenge_fase2.data import load_diagnostic_dataset, split_dataset
from tech_challenge_fase2.models import (
    MODEL_NAMES,
    build_model,
    classification_metrics,
)
from tech_challenge_fase2.optimization import experiment_configs, run_experiment


def _plot_histories(model_name: str, experiments: list[dict[str, Any]], output: Path) -> None:
    plt.figure(figsize=(9, 5))
    for experiment in experiments:
        history = experiment["history"]
        plt.plot(
            [item["generation"] for item in history],
            [item["best"] for item in history],
            marker="o",
            label=experiment["config"]["name"],
        )
    plt.title(f"Evolução do fitness - {model_name}")
    plt.xlabel("Geração")
    plt.ylabel("Fitness (0,65 recall + 0,35 F1)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    dataset = load_diagnostic_dataset(args.data)
    split = split_dataset(dataset)
    result_dir = args.output_dir / "results"
    model_dir = args.output_dir / "models"
    figure_dir = args.output_dir / "figures"
    for directory in (result_dir, model_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "dataset_source": split.source,
        "train_samples": len(split.X_train),
        "test_samples": len(split.X_test),
        "cv_splits": args.cv,
        "fitness_definition": "0.65 * recall + 0.35 * f1",
        "models": {},
    }
    comparison_rows: list[dict[str, Any]] = []

    for model_name in args.models:
        baseline = build_model(model_name)
        baseline.fit(split.X_train, split.y_train)
        baseline_metrics = classification_metrics(baseline, split.X_test, split.y_test)

        serialized_experiments: list[dict[str, Any]] = []
        for config in experiment_configs():
            print(f"[{model_name}] experimento {config.name} iniciado", flush=True)
            started = time.perf_counter()
            result = run_experiment(
                model_name,
                split.X_train,
                split.y_train,
                config,
                cv_splits=args.cv,
            )
            payload = asdict(result)
            payload["duration_seconds"] = round(time.perf_counter() - started, 3)
            serialized_experiments.append(payload)
            experiment_path = result_dir / f"{model_name}_{config.name}.json"
            experiment_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"[{model_name}] {config.name}: fitness={result.best_fitness:.4f}",
                flush=True,
            )

        best = max(serialized_experiments, key=lambda item: item["best_fitness"])
        optimized = build_model(model_name, best["best_parameters"])
        optimized.fit(split.X_train, split.y_train)
        optimized_metrics = classification_metrics(optimized, split.X_test, split.y_test)
        artifact = {
            "model": optimized,
            "model_name": model_name,
            "feature_names": list(split.X_train.columns),
            "parameters": best["best_parameters"],
            "cv_metrics": best["cv_metrics"],
            "test_metrics": optimized_metrics,
            "dataset_source": split.source,
        }
        joblib.dump(artifact, model_dir / f"{model_name}_best.joblib")
        _plot_histories(
            model_name,
            serialized_experiments,
            figure_dir / f"{model_name}_convergence.png",
        )

        summary["models"][model_name] = {
            "baseline_test_metrics": baseline_metrics,
            "optimized_test_metrics": optimized_metrics,
            "selected_experiment": best["config"]["name"],
            "best_parameters": best["best_parameters"],
            "best_cv_metrics": best["cv_metrics"],
            "experiments": serialized_experiments,
        }
        for version, metrics in (("baseline", baseline_metrics), ("optimized", optimized_metrics)):
            comparison_rows.append(
                {
                    "model": model_name,
                    "version": version,
                    **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                    "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                }
            )

    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(comparison_rows).to_csv(result_dir / "comparison.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

