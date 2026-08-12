#!/usr/bin/env python3
"""Execute the small, currently compatible notebook smoke set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


SMOKE_NOTEBOOKS = (
    Path("EM/em.ipynb"),
    Path("KNearestNeighbors/KNN.ipynb"),
    Path("LeastSquaresMethod/least_sqaure_method.ipynb"),
    Path("NaiveBayes/GaussianNB.ipynb"),
    Path("Perceptron/Iris_perceptron.ipynb"),
    Path("SVM/support-vector-machine.ipynb"),
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute notebooks that are currently compatible with the maintenance baseline."
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help="Notebook paths relative to the repository root; defaults to the smoke set.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Maximum seconds allowed for one notebook cell (default: 120).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repository_root()
    notebook_paths = tuple(args.notebooks) if args.notebooks else SMOKE_NOTEBOOKS

    for relative_path in notebook_paths:
        path = root / relative_path
        if not path.is_file():
            print(f"Notebook not found: {relative_path}", file=sys.stderr)
            return 2

    for relative_path in notebook_paths:
        path = root / relative_path
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            allow_errors=False,
            resources={"metadata": {"path": str(root)}},
        ).execute()
        print(f"Executed {relative_path}")

    print(f"Executed {len(notebook_paths)} notebook(s) successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
