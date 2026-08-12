# 2026 issue and pull-request triage

Triage date: 12 August 2026.

## Repository backlog status

GitHub Issues are disabled for this repository, so there is no issue backlog to
review or label. The initial audit found no existing pull-request backlog. The
public maintenance record is the stacked PR chain below and its final master
promotion.

The repository API reports an open issue-like count that includes pull
requests; it must not be interpreted as standalone GitHub Issues while the
Issues feature is disabled.

## Completed PR triage

PRs #9–#19 were marked ready and merged in dependency order. Each had passing
Python 3.10, 3.11, 3.12, and 3.13 checks for both the push and pull-request
workflow events.

| PR | Area | Triage result | Final disposition |
| --- | --- | --- | --- |
| [#9](https://github.com/wzyonggege/statistical-learning-method/pull/9) | Foundation | Ready; parent of the stack | Merged into `master` |
| [#10](https://github.com/wzyonggege/statistical-learning-method/pull/10) | Least squares | Ready; depended on #9 | Merged into the maintenance stack |
| [#11](https://github.com/wzyonggege/statistical-learning-method/pull/11) | SVM | Ready; depended on #10 | Merged into the maintenance stack |
| [#12](https://github.com/wzyonggege/statistical-learning-method/pull/12) | Perceptron | Ready; depended on #11 | Merged into the maintenance stack |
| [#13](https://github.com/wzyonggege/statistical-learning-method/pull/13) | k-nearest neighbors | Ready; depended on #12 | Merged into the maintenance stack |
| [#14](https://github.com/wzyonggege/statistical-learning-method/pull/14) | Naive Bayes | Ready; depended on #13 | Merged into the maintenance stack |
| [#15](https://github.com/wzyonggege/statistical-learning-method/pull/15) | Decision tree | Ready; depended on #14 | Merged into the maintenance stack |
| [#16](https://github.com/wzyonggege/statistical-learning-method/pull/16) | Logistic regression | Ready; depended on #15 | Merged into the maintenance stack |
| [#17](https://github.com/wzyonggege/statistical-learning-method/pull/17) | AdaBoost | Ready; depended on #16 | Merged into the maintenance stack |
| [#18](https://github.com/wzyonggege/statistical-learning-method/pull/18) | EM | Ready; depended on #17 | Merged into the maintenance stack |
| #19 | Release readiness | Ready; depended on #18 | Merged into the maintenance stack |
| Final | Master promotion | Completed after #19 | Promoted to `master`; release tag pending |

No PR was closed as a duplicate. Future stacked changes should retain the
parent-first merge workflow and the same CI evidence.

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
