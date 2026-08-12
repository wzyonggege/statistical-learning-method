# Statistical Learning Method

> 《统计学习方法》笔记——基于 Python 算法实现

This repository is an educational companion to classic statistical-learning
methods. It explains the mathematics, implements the main ideas by hand in
Python, and uses scikit-learn only for comparison cells that were part of the
original notebooks.

The 2026 revival keeps that original purpose intact: the notebooks are learning
material first, and a modern runtime is a way to make that material usable
again—not a reason to replace the implementations with library calls.

## Project history and scope

The project was created on 28 December 2017 and its original development
series ended on 9 January 2018. The original Chinese notes, examples, plots,
and chapter order are preserved while the maintenance work adds reproducible
setup, validation, and small compatibility fixes in separate pull requests.

## Nine chapters

| Chapter | Topic | Notebook |
| --- | --- | --- |
| 1 | 最小二乘法 / Least squares | [least_sqaure_method.ipynb](LeastSquaresMethod/least_sqaure_method.ipynb) |
| 2 | 感知机 / Perceptron | [Iris_perceptron.ipynb](Perceptron/Iris_perceptron.ipynb) |
| 3 | k 近邻法 / k-nearest neighbors | [KNN.ipynb](KNearestNeighbors/KNN.ipynb) |
| 4 | 朴素贝叶斯 / Naive Bayes | [GaussianNB.ipynb](NaiveBayes/GaussianNB.ipynb) |
| 5 | 决策树 / Decision tree | [DT.ipynb](DecisonTree/DT.ipynb) |
| 6 | 逻辑斯谛回归 / Logistic regression | [LR.ipynb](LogisticRegression/LR.ipynb) |
| 7 | 支持向量机 / SVM | [support-vector-machine.ipynb](SVM/support-vector-machine.ipynb) |
| 8 | AdaBoost | [Adaboost.ipynb](AdaBoost/Adaboost.ipynb) |
| 9 | EM 算法 / Expectation-Maximization | [em.ipynb](EM/em.ipynb) |

## Quick start

The supported development range is Python 3.10–3.13. [uv](https://docs.astral.sh/uv/)
creates the environment from the committed lock file:

```bash
uv sync --all-groups
uv run python -m pytest
uv run python scripts/validate_notebooks.py
uv run python scripts/execute_notebooks.py
```

The last command executes the small smoke set that is currently compatible
with modern dependencies. The validator still checks every committed notebook.
Open the notebooks in JupyterLab with:

```bash
uv run jupyter lab
```

The decision-tree notebook's final visualization also needs the system
Graphviz `dot` executable. The Python `graphviz` package is included in the
environment; install the executable with your platform's package manager when
you want to render that cell.

## 2026 maintenance status

**Branch:** `maintenance/2026-revival`<br>
**Current stage:** PR 3 — SVM modernization and correctness tests<br>
**Original upstream baseline:** last commit 9 January 2018; no published release

### Compatibility audit snapshot

The following snapshot was collected on 12 August 2026 in a clean CPython
3.13.13 virtual environment using the latest versions resolved for this audit.
The committed `uv.lock` is the reproducibility source; this table records what
was actually exercised during the baseline audit.

| Component | Audit version |
| --- | --- |
| Python | 3.13.13 |
| NumPy | 2.5.2 |
| pandas | 3.0.5 |
| matplotlib | 3.11.1 |
| scikit-learn | 1.9.0 |

| Notebook | Baseline result | Finding / follow-up |
| --- | --- | --- |
| Least squares | Pass | Smoke-tested; preserve the curve-fitting explanation and outputs. PR 2. |
| Perceptron | Partial | Hand-written cells run; sklearn comparison still uses removed `n_iter`. PR 4. |
| k-nearest neighbors | Pass | Smoke-tested; add deterministic correctness tests in PR 5. |
| Naive Bayes | Partial | Hand-written cells run; sklearn prediction needs a 2-D single-sample input. PR 6. |
| Decision tree | Partial | Learning cells run; final display needs the system `dot` executable. PR 7. |
| Logistic regression | Partial | Hand-written `fit` passes a 1-D NumPy array to `math.exp`. PR 8. |
| SVM | Pass | `model_selection.train_test_split` migration complete; hand-written SMO module and tests added in PR 3. |
| AdaBoost | Pass, slow | Executes but takes about 2.5 minutes in the audit environment; keep out of the smoke set until PR 9. |
| EM | Pass | Smoke-tested; add convergence/correctness coverage in PR 10. |

The baseline notebook metadata still identifies Python 3.6.1. That metadata is
historical, not a supported runtime declaration. No notebook implementation
behavior is changed in PR 1.

### CI policy

GitHub Actions validates all nine primary notebooks as nbformat 4 documents and
executes the four current smoke notebooks on Python 3.10, 3.11, 3.12, and
3.13. The remaining notebooks are named above rather than hidden behind a
green-but-meaningless `allow_errors` execution.

### Planned pull-request sequence

1. **PR 1 — Foundation:** MIT license, ignore rules,
   checkpoint cleanup, locked dependencies, notebook validation/smoke CI,
   README, and contribution guidance.
2. **PR 2 — Least squares:** modern execution boundaries and lightweight
   numerical correctness tests.
3. **PR 3 — SVM:** migrate deprecated imports, preserve the hand-written SMO
   explanation, and add kernel/classification checks.
4. **PR 4 — Perceptron:** current sklearn comparison API and hand-written
   classifier tests.
5. **PR 5 — k-nearest neighbors:** input validation, deterministic examples,
   and distance/prediction tests.
6. **PR 6 — Naive Bayes:** modern single-sample comparison shape and tests for
   Gaussian probability/classification behavior.
7. **PR 7 — Decision tree:** preserve the entropy/information-gain implementation,
   make the visualization path portable, and test tree predictions.
8. **PR 8 — Logistic regression:** preserve gradient descent while making
   scalar/array behavior explicit and testing convergence on the teaching data.
9. **PR 9 — AdaBoost:** retain the threshold weak learners, address runtime and
   edge cases, and test weight updates/classification.
10. **PR 10 — EM:** preserve the generator-based teaching flow and add
    parameter/convergence tests.
11. **PR 11 — Release readiness:** CHANGELOG, 2026 revival release plan,
    final compatibility matrix, and issue/PR triage record.

At audit time GitHub Issues are disabled and the repository reports zero pull
requests, so there is no existing issue or PR backlog to merge into this plan.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow, notebook
preservation rules, validation commands, and pull-request boundaries.

## License

This project is released under the [MIT License](LICENSE), copyright
`wzyonggege`.
