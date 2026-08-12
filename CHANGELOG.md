# Changelog

All notable changes to this educational repository are documented here.

## Unreleased — 2026 revival

This work is not released yet. It is being reviewed as a stack of Draft pull
requests and must not be described as merged until the maintainer approves the
merge order.

### Repository foundation

- Added the MIT License with copyright holder `wzyonggege`.
- Added Python, notebook, build-artifact, and editor ignore rules; removed
  committed `.ipynb_checkpoints` content.
- Added reproducible `pyproject.toml` and `uv.lock` dependency management for
  Python 3.10–3.13.
- Added notebook structure validation, smoke execution, and a four-version
  GitHub Actions matrix.
- Rewrote the README and added contribution and maintenance guidance.

### Hand-written algorithm chapters

- Least squares: added a tested reusable implementation while preserving the
  original curve-fitting notebook.
- SVM: migrated the deprecated model-selection import and added tests for the
  hand-written SMO and kernels.
- Perceptron: extracted the stochastic-gradient implementation and added
  deterministic classifier tests.
- k-nearest neighbors: extracted distance/voting logic and added prediction
  and accuracy tests.
- Naive Bayes: extracted the hand-written Gaussian classifier and fixed the
  single-sample comparison shape.
- Decision tree: extracted the entropy/information-gain ID3 implementation,
  kept the historical import path, and made Graphviz rendering optional.
- Logistic regression: replaced scalar-only `math.exp` use with a NumPy-safe
  sigmoid while preserving the stochastic-gradient update.
- AdaBoost: retained threshold weak learners and added finite perfect-stump,
  constant-feature, and coarse-threshold handling.
- EM: retained the generator-based `next`/`send` teaching flow and removed its
  dependency on module-level observation state.

### Verification

- Added lightweight correctness tests for all nine chapters.
- The current stack validates all nine notebooks and executes all nine smoke
  notebooks successfully.
- The GitHub Actions checks for Python 3.10, 3.11, 3.12, and 3.13 pass on the
  current Draft PR chain.

### Compatibility notes

- The locked audit environment records NumPy 2.5.2, pandas 3.0.5, matplotlib
  3.11.1, and scikit-learn 1.9.0 on CPython 3.13.13.
- Decision-tree image rendering additionally needs the system Graphviz `dot`
  executable; notebook execution does not.
- Original notebook metadata and educational explanations remain historical
  material rather than runtime declarations.

## Release policy

The first revival release will be tagged only after the stacked Draft PRs are
reviewed, merged in order, and the final release gates in
[`docs/2026-revival-release-plan.md`](docs/2026-revival-release-plan.md) pass.
