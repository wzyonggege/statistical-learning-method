# Contributing

Thank you for helping keep these statistical-learning notes useful to learners.
The repository is intentionally educational: the hand-written implementations
and the mathematical explanations are the primary material. scikit-learn is
used only as a comparison where the original notebook already included it.

## Development setup

The supported development range is Python 3.10–3.13. Install the locked
development environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-groups
```

The decision-tree visualization also needs the Graphviz `dot` executable when
that notebook is run through its final display cell. The Python `graphviz`
package is included in the locked environment; install the system executable
using your platform's package manager when you need that visualization.

Before opening a pull request, run:

```bash
uv run python scripts/validate_notebooks.py
uv run python scripts/execute_notebooks.py
```

The smoke command intentionally executes only the notebooks currently known to
run on the maintenance baseline. The compatibility table in the README names
the remaining notebooks and their planned maintenance PR.

## Notebook and algorithm changes

- Read the complete notebook, including its mathematical explanation and
  existing outputs, before changing implementation behavior.
- Keep the hand-written algorithm implementation. Do not replace it with a
  scikit-learn estimator; comparison cells may remain as comparisons.
- Keep examples deterministic where practical and explain any changed output
  or random seed in the notebook.
- Add a lightweight correctness test for each algorithm change.
- Do not commit `.ipynb_checkpoints`, local virtual environments, generated
  caches, or machine-specific files.

## Pull requests

Use a dedicated branch and keep each pull request logically scoped. The 2026
revival is intentionally split into a foundation PR, one PR per algorithm
chapter, and a final release-documentation/triage PR. A PR description should
include:

1. the problem or compatibility failure;
2. the maintenance approach and why the educational behavior is preserved;
3. the exact validation commands and their results;
4. any remaining limitation or follow-up PR.

Please do not push directly to `master` or merge a maintenance PR without
maintainer approval.
