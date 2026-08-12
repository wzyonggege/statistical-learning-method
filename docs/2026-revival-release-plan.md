# 2026 revival release plan

Status as of 12 August 2026: implementation work is complete in the local
maintenance stack, but the changes are still public Draft PRs and are not yet
merged or released.

## Release objective

Publish the first maintained release of the 2017–2018 educational project with
the original mathematics and hand-written algorithms intact, a reproducible
modern Python environment, executable notebooks, and lightweight correctness
coverage.

The planned tag is `v2026.1.0`. It is a plan only; no tag is created by this
Draft PR.

## Public work sequence

The revival sequence is intentionally stacked so every change has a reviewable
scope:

| Revival step | GitHub PR | Scope | Merge prerequisite |
| --- | --- | --- | --- |
| 1 | [#9](https://github.com/wzyonggege/statistical-learning-method/pull/9) | Foundation, dependencies, CI, README | none; review against `master` |
| 2 | [#10](https://github.com/wzyonggege/statistical-learning-method/pull/10) | Least squares | #9 |
| 3 | [#11](https://github.com/wzyonggege/statistical-learning-method/pull/11) | SVM | #10 |
| 4 | [#12](https://github.com/wzyonggege/statistical-learning-method/pull/12) | Perceptron | #11 |
| 5 | [#13](https://github.com/wzyonggege/statistical-learning-method/pull/13) | k-nearest neighbors | #12 |
| 6 | [#14](https://github.com/wzyonggege/statistical-learning-method/pull/14) | Naive Bayes | #13 |
| 7 | [#15](https://github.com/wzyonggege/statistical-learning-method/pull/15) | Decision tree | #14 |
| 8 | [#16](https://github.com/wzyonggege/statistical-learning-method/pull/16) | Logistic regression | #15 |
| 9 | [#17](https://github.com/wzyonggege/statistical-learning-method/pull/17) | AdaBoost | #16 |
| 10 | [#18](https://github.com/wzyonggege/statistical-learning-method/pull/18) | EM | #17 |
| 11 | This PR | Changelog, release plan, and triage record | #18 |

All algorithm PRs are Draft and all recorded CI jobs pass. Draft status is
intentional: approval is still required before any merge.

## Release gates

Before tagging `v2026.1.0`, the maintainer should verify:

1. Approve and merge PRs #9–#18 in order; do not merge a child branch before
   its parent is accepted.
2. Review the final diff for preservation of mathematical explanations,
   notebook outputs where meaningful, and hand-written implementations.
3. Require the Python 3.10–3.13 GitHub Actions matrix to remain green after
   the final merge result.
4. Run the locked local checks from the repository root:

   ```bash
   uv run --locked --all-groups pytest -q
   uv run --locked --all-groups python scripts/validate_notebooks.py
   uv run --locked --all-groups python scripts/execute_notebooks.py
   uv lock --check
   ```

5. Confirm there are no committed checkpoints, stored notebook errors, or
   generated machine-specific files.
6. Update `CHANGELOG.md` from “Unreleased” to the release date, record the
   final commit, create the tag, and publish release notes only after approval.

## Post-release maintenance

- Keep the four-version CI matrix and run it for future notebook changes.
- Keep one logically scoped PR per algorithm or maintenance concern.
- Use GitHub PRs for the current triage trail while Issues remain disabled.
- Re-audit dependencies when Python, NumPy, pandas, matplotlib, or
  scikit-learn major versions change.
- Document any notebook output change instead of silently rewriting history.
