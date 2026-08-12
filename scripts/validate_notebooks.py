#!/usr/bin/env python3
"""Validate every committed notebook without executing it."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    root = repository_root()
    notebooks = sorted(root.rglob("*.ipynb"))
    checkpoint_paths = [
        path for path in notebooks if ".ipynb_checkpoints" in path.parts
    ]

    failures: list[str] = []
    for path in checkpoint_paths:
        failures.append(f"checkpoint notebook must not be committed: {path.relative_to(root)}")

    for path in notebooks:
        if path in checkpoint_paths:
            continue
        try:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell_number, cell in enumerate(notebook.cells):
                for output in cell.get("outputs", []):
                    if output.get("output_type") == "error":
                        failures.append(
                            f"stored error output in {path.relative_to(root)} cell {cell_number}"
                        )
        except Exception as exc:  # pragma: no cover - exercised by malformed files
            failures.append(f"{path.relative_to(root)}: {type(exc).__name__}: {exc}")

    if failures:
        print("Notebook validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"Validated {len(notebooks)} notebook(s); no checkpoint or stored error output found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
