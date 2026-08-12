# 2026 issue and pull-request triage

Triage date: 12 August 2026.

## Repository backlog status

GitHub Issues are disabled for this repository, so there is no issue backlog to
review or label. The initial audit found no existing pull-request backlog. The
public maintenance record is now the stacked Draft PR chain below.

The repository API reports an open issue-like count that includes pull
requests; it must not be interpreted as standalone GitHub Issues while the
Issues feature is disabled.

## Open PR triage

Every PR below is `OPEN` and `DRAFT`. Each has passing Python 3.10, 3.11,
3.12, and 3.13 checks for both the push and pull-request workflow events.

| PR | Area | Triage result | Next action |
| --- | --- | --- | --- |
| [#9](https://github.com/wzyonggege/statistical-learning-method/pull/9) | Foundation | Ready for maintainer review; parent of the stack | Review and merge first if approved |
| [#10](https://github.com/wzyonggege/statistical-learning-method/pull/10) | Least squares | Ready; depends on #9 | Review after #9 |
| [#11](https://github.com/wzyonggege/statistical-learning-method/pull/11) | SVM | Ready; depends on #10 | Review after #10 |
| [#12](https://github.com/wzyonggege/statistical-learning-method/pull/12) | Perceptron | Ready; depends on #11 | Review after #11 |
| [#13](https://github.com/wzyonggege/statistical-learning-method/pull/13) | k-nearest neighbors | Ready; depends on #12 | Review after #12 |
| [#14](https://github.com/wzyonggege/statistical-learning-method/pull/14) | Naive Bayes | Ready; depends on #13 | Review after #13 |
| [#15](https://github.com/wzyonggege/statistical-learning-method/pull/15) | Decision tree | Ready; depends on #14 | Review after #14 |
| [#16](https://github.com/wzyonggege/statistical-learning-method/pull/16) | Logistic regression | Ready; depends on #15 | Review after #15 |
| [#17](https://github.com/wzyonggege/statistical-learning-method/pull/17) | AdaBoost | Ready; depends on #16 | Review after #16 |
| [#18](https://github.com/wzyonggege/statistical-learning-method/pull/18) | EM | Ready; depends on #17 | Review after #17 |

No PR is marked for closure, duplication, or merge without review. The stack
should be rebased or retargeted only as part of an approved merge workflow.

## Triage policy for new reports

Until Issues are enabled, new maintenance reports should be opened as focused
GitHub Draft PRs or discussed on the relevant existing PR. A report should
include:

- the exact notebook or module and reproduction command;
- the Python/dependency versions;
- expected versus observed mathematical or execution behavior;
- a minimal test or notebook cell when possible; and
- whether the report changes educational behavior or only portability.

Security or data-loss reports should not be bundled into an algorithm PR; they
need a separate review path and explicit maintainer attention.
