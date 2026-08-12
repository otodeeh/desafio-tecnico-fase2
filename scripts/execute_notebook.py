"""Executa um notebook do projeto e salva os outputs no próprio arquivo."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = PROJECT_ROOT / "notebooks" / "tech_challenge_fase2_completo.ipynb"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", nargs="?", type=Path, default=DEFAULT_NOTEBOOK)
    args = parser.parse_args()
    path = args.notebook.resolve()

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="tech-challenge-fase2",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, path)
    print(f"Notebook executado e salvo em {path}")


if __name__ == "__main__":
    main()

