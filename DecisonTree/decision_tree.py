"""Educational ID3 decision tree using entropy and information gain."""

from __future__ import annotations

from math import log

import numpy as np
import pandas as pd


def create_data() -> tuple[list[list[str]], list[str]]:
    """Return the categorical dataset from the textbook example 5.1."""

    datasets = [
        ["青年", "否", "否", "一般", "否"],
        ["青年", "否", "否", "好", "否"],
        ["青年", "是", "否", "好", "是"],
        ["青年", "是", "是", "一般", "是"],
        ["青年", "否", "否", "一般", "否"],
        ["中年", "否", "否", "一般", "否"],
        ["中年", "否", "否", "好", "否"],
        ["中年", "是", "是", "好", "是"],
        ["中年", "否", "是", "非常好", "是"],
        ["中年", "否", "是", "非常好", "是"],
        ["老年", "否", "是", "非常好", "是"],
        ["老年", "否", "是", "好", "是"],
        ["老年", "是", "否", "好", "是"],
        ["老年", "是", "否", "非常好", "是"],
        ["老年", "否", "否", "一般", "否"],
    ]
    labels = ["年龄", "有工作", "有自己的房子", "信贷情况", "类别"]
    return datasets, labels


class Node:
    """A node in the categorical ID3 tree."""

    def __init__(
        self,
        root: bool = True,
        label: object | None = None,
        feature_name: object | None = None,
        feature: int | None = None,
    ) -> None:
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree: dict[object, Node] = {}
        self.result = {"label:": self.label, "feature": self.feature, "tree": self.tree}

    def __repr__(self) -> str:
        return str(self.result)

    def add_node(self, value: object, node: "Node") -> None:
        self.tree[value] = node

    def predict(self, features: list[object] | tuple[object, ...]) -> object:
        if self.root:
            return self.label
        if self.feature is None:
            raise RuntimeError("non-leaf node has no split feature")
        value = features[self.feature]
        try:
            child = self.tree[value]
        except KeyError as exc:
            raise ValueError(
                f"no decision-tree branch for feature value {value!r}"
            ) from exc
        return child.predict(features)


class DTree:
    """ID3 classifier following the notebook's entropy derivation."""

    def __init__(self, epsilon: float = 0.1) -> None:
        self.epsilon = epsilon
        self._tree: Node | None = None

    @staticmethod
    def calc_ent(datasets: list[list[object]] | np.ndarray) -> float:
        data_length = len(datasets)
        if data_length == 0:
            raise ValueError("datasets must not be empty")

        label_count: dict[object, int] = {}
        for row in datasets:
            label = row[-1]
            label_count[label] = label_count.get(label, 0) + 1
        return -sum(
            (count / data_length) * log(count / data_length, 2)
            for count in label_count.values()
        )

    def cond_ent(
        self, datasets: list[list[object]] | np.ndarray, axis: int = 0
    ) -> float:
        data_length = len(datasets)
        if data_length == 0:
            raise ValueError("datasets must not be empty")

        feature_sets: dict[object, list[list[object]] | list[np.ndarray]] = {}
        for row in datasets:
            feature_sets.setdefault(row[axis], []).append(row)
        return sum(
            (len(rows) / data_length) * self.calc_ent(rows)
            for rows in feature_sets.values()
        )

    @staticmethod
    def info_gain(ent: float, cond_ent: float) -> float:
        return ent - cond_ent

    def info_gain_train(self, datasets: np.ndarray) -> tuple[int, float]:
        if datasets.ndim != 2 or len(datasets) == 0 or datasets.shape[1] < 2:
            raise ValueError("datasets must have rows and at least one feature")

        feature_count = datasets.shape[1] - 1
        ent = self.calc_ent(datasets)
        gains = [
            (feature, self.info_gain(ent, self.cond_ent(datasets, axis=feature)))
            for feature in range(feature_count)
        ]
        return max(gains, key=lambda item: item[1])

    def train(self, train_data: pd.DataFrame) -> Node:
        """Recursively build an ID3 tree from a labeled DataFrame."""

        if not isinstance(train_data, pd.DataFrame) or train_data.empty:
            raise ValueError("train_data must be a non-empty pandas DataFrame")
        if len(train_data.columns) < 1:
            raise ValueError("train_data must include a label column")

        labels = train_data.iloc[:, -1]
        features = list(train_data.columns[:-1])

        if labels.nunique() == 1:
            return Node(root=True, label=labels.iloc[0])

        if not features:
            return Node(root=True, label=labels.value_counts().index[0])

        max_feature, max_info_gain = self.info_gain_train(train_data.to_numpy())
        max_feature_name = features[max_feature]

        if max_info_gain < self.epsilon:
            return Node(root=True, label=labels.value_counts().index[0])

        node_tree = Node(
            root=False,
            feature_name=max_feature_name,
            feature=max_feature,
        )
        for value in train_data[max_feature_name].value_counts().index:
            subset = train_data.loc[
                train_data[max_feature_name] == value
            ].drop(columns=[max_feature_name])
            node_tree.add_node(value, self.train(subset))
        return node_tree

    def fit(self, train_data: pd.DataFrame) -> Node:
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, features: list[object] | tuple[object, ...]) -> object:
        if self._tree is None:
            raise RuntimeError("fit the decision tree before predicting")
        return self._tree.predict(features)


__all__ = ["DTree", "Node", "create_data"]
