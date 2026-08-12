import numpy as np
import pandas as pd

from DecisonTree.decision_tree import DTree, create_data


def test_id3_entropy_is_one_for_two_equally_likely_classes():
    datasets = [["a", "否"], ["b", "是"]]

    assert DTree.calc_ent(datasets) == 1.0


def test_id3_selects_house_ownership_as_the_first_split():
    datasets, _ = create_data()

    feature, gain = DTree().info_gain_train(np.array(datasets))

    assert feature == 2
    assert gain > 0.3


def test_id3_predicts_the_original_textbook_example():
    datasets, labels = create_data()
    model = DTree()

    model.fit(pd.DataFrame(datasets, columns=labels))

    assert model.predict(["老年", "否", "否", "一般"]) == "否"


def test_legacy_dt_module_reexports_the_shared_implementation():
    from DecisonTree.dt import DTree as LegacyDTree

    assert LegacyDTree is DTree
