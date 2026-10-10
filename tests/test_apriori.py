import pandas as pd
import pytest

from src.apriori_engine import (apriori_native, generate_candidates, mine_frequent_itemsets, trace_to_frame,
                                validate_threshold)
from src.pipeline import prepare_data
from tests.helpers import PRUNE_BASKETS, fs, onehot


def as_dict(itemsets: pd.DataFrame, column: str = "support") -> dict:
    return dict(zip(itemsets["itemset"], itemsets[column]))


def test_supports_match_hand_calculation():
    found = as_dict(apriori_native(onehot(), 0.4))
    expected = {fs("A"): 0.8, fs("B"): 0.8, fs("C"): 0.8, fs("AB"): 0.6, fs("AC"): 0.6, fs("BC"): 0.6, fs("ABC"): 0.4}
    assert set(found) == set(expected)
    for k, v in expected.items():
        assert found[k] == pytest.approx(v)


def test_counts_and_sizes():
    out = apriori_native(onehot(), 0.4)
    assert as_dict(out, "count")[fs("ABC")] == 2
    assert out["size"].value_counts().to_dict() == {1: 3, 2: 3, 3: 1}
    assert ((out["support"] > 0) & (out["support"] <= 1)).all()


def test_threshold_is_inclusive_and_monotone():
    oh = onehot()
    assert fs("ABC") in as_dict(apriori_native(oh, 0.4))      # support exactly 0.4
    assert fs("ABC") not in as_dict(apriori_native(oh, 0.41))
    sizes = [len(apriori_native(oh, s)) for s in (0.2, 0.4, 0.6, 0.8, 1.0)]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] == 15 and sizes[-1] == 0                 # all 2^4-1 subsets vs none


def test_no_frequent_itemsets_returns_empty_frame():
    out = apriori_native(onehot(), 1.0)
    assert out.empty and list(out.columns) == ["itemset", "size", "support", "count"]


def test_candidate_generation_join_and_prune():
    kept, pruned = generate_candidates([("A", "B"), ("A", "C"), ("B", "C")])
    assert kept == [("A", "B", "C")] and pruned == []
    kept, pruned = generate_candidates([("A", "B"), ("A", "C")])
    assert kept == [] and pruned == [("A", "B", "C")]         # {B, C} is not frequent
    assert generate_candidates([("A",), ("B",), ("C",)])[0] == [("A", "B"), ("A", "C"), ("B", "C")]


def test_trace_shows_subset_pruning():
    oh = onehot(PRUNE_BASKETS)
    itemsets, trace = apriori_native(oh, 0.4, return_trace=True)
    assert set(itemsets["itemset"]) == {fs("A"), fs("B"), fs("C"), fs("AB"), fs("AC")}
    level2, level3 = trace[1], trace[2]
    assert level2["pruned_by_support"] == [("B", "C")] and level2["counted"][("B", "C")] == 1
    assert level3["pruned_by_subset"] == [("A", "B", "C")] and level3["counted"] == {}
    frame = trace_to_frame(trace, len(oh), 0.4)
    assert frame["outcome"].str.contains("infrequent subset").sum() == 1


def test_apriori_property_holds_on_project_data():
    itemsets = apriori_native(prepare_data().onehot, 0.02)
    support = as_dict(itemsets)
    assert itemsets["size"].max() >= 3
    for itemset, sup in support.items():
        for item in itemset:
            subset = itemset - {item}
            if subset:
                assert subset in support and support[subset] >= sup - 1e-12


def test_max_len():
    assert apriori_native(onehot(), 0.4, max_len=2)["size"].max() == 2


@pytest.mark.parametrize("bad", [0, -0.1, 1.5, "0.3", None, float("nan"), True])
def test_invalid_min_support(bad):
    with pytest.raises(ValueError):
        apriori_native(onehot(), bad)


def test_threshold_boundaries():
    assert validate_threshold(1, "x") == 1.0
    assert validate_threshold(0, "x", allow_zero=True) == 0.0
    with pytest.raises(ValueError):
        validate_threshold(0, "x")


def test_unknown_backend_and_empty_matrix():
    with pytest.raises(ValueError):
        mine_frequent_itemsets(onehot(), 0.4, backend="eclat")
    with pytest.raises(ValueError):
        apriori_native(pd.DataFrame(), 0.4)


def test_native_matches_mlxtend():
    """Cross-check against the reference library (skipped if mlxtend is not installed)."""
    pytest.importorskip("mlxtend")
    oh = prepare_data().onehot
    for s in (0.02, 0.05, 0.1):
        native = as_dict(mine_frequent_itemsets(oh, s, backend="native"))
        reference = as_dict(mine_frequent_itemsets(oh, s, backend="mlxtend"))
        assert set(native) == set(reference)
        for k in native:
            assert native[k] == pytest.approx(reference[k])
