"""Materializa uma cópia local normalizada do dataset usado nos experimentos."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tech_challenge_fase2.data import load_diagnostic_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/raw/data.csv"))
    args = parser.parse_args()
    dataset = load_diagnostic_dataset(args.source)
    output = dataset.features.copy()
    output.insert(0, "diagnosis", dataset.target.map({1: "M", 0: "B"}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Dataset preparado em {args.output} a partir de {dataset.source}.")


if __name__ == "__main__":
    main()

