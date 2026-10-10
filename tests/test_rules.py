import pytest

from src.apriori_engine import apriori_native
from src.evaluation import item_coverage, threshold_experiment
from src.rule_generation import generate_rules, rules_for_display, search_rules
from src.visualizations import build_rule_graph, component_layout
from tests.helpers import fs, onehot


def rules_at(min_support=0.4, min_confidence=0.0, baskets=None):
    oh = onehot(baskets) if baskets else onehot()
    return generate_rules(apriori_native(oh, min_support), min_confidence)


def get(rules, a, b):
    hit = rules[(rules["antecedent"] == fs(a)) & (rules["consequent"] == fs(b))]
    assert len(hit) == 1, f"rule {a}->{b} not found exactly once"
    return hit.iloc[0]


def test_confidence_matches_hand_calculation():
    rules = rules_at()
    assert get(rules, "A", "B")["confidence"] == pytest.approx(3 / 4)       # 0.6 / 0.8
    assert get(rules, "AB", "C")["confidence"] == pytest.approx(2 / 3)      # 0.4 / 0.6
    assert get(rules, "A", "BC")["confidence"] == pytest.approx(1 / 2)      # 0.4 / 0.8
    assert get(rules, "AB", "C")["support"] == pytest.approx(0.4)
    assert get(rules, "AB", "C")["count"] == 2


def test_rule_count_is_complete():
    # 3 pairs x 2 directions + the triple's 6 antecedent/consequent splits
    assert len(rules_at()) == 12


def test_reverse_rules_have_same_support_but_different_confidence():
    baskets = {"T1": "XY", "T2": "XY", "T3": "X", "T4": "X", "T5": "Z"}
    rules = rules_at(0.4, 0.0, baskets)
    xy, yx = get(rules, "X", "Y"), get(rules, "Y", "X")
    assert xy["support"] == pytest.approx(0.4) and yx["support"] == pytest.approx(0.4)
    assert xy["confidence"] == pytest.approx(0.5)     # 2 of the 4 baskets with X
    assert yx["confidence"] == pytest.approx(1.0)     # 2 of the 2 baskets with Y


def test_min_confidence_filters_and_is_monotone():
    counts = [len(rules_at(0.4, c)) for c in (0.0, 0.5, 0.6, 0.7, 0.8, 1.0)]
    assert counts == sorted(counts, reverse=True)
    assert counts == [12, 12, 9, 6, 0, 0]
    assert (rules_at(0.4, 0.7)["confidence"] >= 0.7).all()


def test_empty_inputs_give_empty_rules():
    assert generate_rules(apriori_native(onehot(), 1.0), 0.5).empty
    only_singletons = apriori_native(onehot(), 0.7)   # A, B, C only
    assert set(only_singletons["size"]) == {1}
    out = generate_rules(only_singletons, 0.1)
    assert out.empty and "confidence" in out.columns


@pytest.mark.parametrize("bad", [-0.01, 1.01, "high", None, float("nan")])
def test_invalid_min_confidence(bad):
    with pytest.raises(ValueError):
        generate_rules(apriori_native(onehot(), 0.4), bad)


def test_rule_search():
    rules = rules_at()
    assert len(search_rules(rules, items=["A"])) == 10            # every rule except B->C and C->B
    assert len(search_rules(rules, items=["A"], where="antecedent")) == 5
    assert len(search_rules(rules, items=["A"], where="consequent")) == 5
    assert len(search_rules(rules, items=["A", "B", "C"], match_all=True)) == 6
    assert len(search_rules(rules, text="a")) == 10               # case-insensitive
    assert search_rules(rules, items=["Z"]).empty
    assert len(search_rules(rules)) == 12
    with pytest.raises(ValueError):
        search_rules(rules, where="left")


def test_display_frame():
    shown = rules_for_display(rules_at())
    assert list(shown.columns)[:4] == ["Antecedent (LHS)", "Consequent (RHS)", "Support", "Confidence"]
    assert rules_for_display(rules_at(1.0)).empty


def test_rule_graph_edges_come_from_rules():
    rules = rules_at(0.4, 0.7)        # A->B, B->A, A->C, C->A, B->C, C->B at 0.75
    graph = build_rule_graph(rules, max_rules=50, item_support={"A": 0.8})
    assert set(graph.nodes) == {"A", "B", "C"} and graph.number_of_edges() == 6
    assert graph["A"]["B"]["confidence"] == pytest.approx(0.75)
    assert graph.nodes["A"]["support"] == 0.8
    assert build_rule_graph(rules_at(1.0)).number_of_nodes() == 0
    assert build_rule_graph(rules, max_rules=1).number_of_edges() == 1


def test_multi_item_rule_expands_to_item_edges():
    rules = rules_at(0.4, 0.0)
    only = rules[(rules["antecedent"] == fs("AB")) & (rules["consequent"] == fs("C"))]
    graph = build_rule_graph(only)
    assert set(graph.edges) == {("A", "C"), ("B", "C")}


def test_threshold_experiment_and_coverage():
    table = threshold_experiment(onehot(), [0.4, 0.6], [0.5, 0.8], basket=["A"])
    assert len(table) == 4
    row = table[(table["min_support"] == 0.4) & (table["min_confidence"] == 0.5)].iloc[0]
    assert row["frequent_itemsets"] == 7 and row["rules"] == 12
    high = table[(table["min_support"] == 0.6) & (table["min_confidence"] == 0.8)].iloc[0]
    assert high["frequent_itemsets"] == 6 and high["rules"] == 0 and high["example_recommendations"] == "none"
    assert item_coverage(rules_at(0.4, 0.7), list("ABCD")) == pytest.approx(0.75)


def test_component_layout_separates_clusters():
    import networkx as nx

    graph = nx.DiGraph([("A", "B"), ("B", "C"), ("C", "A"), ("X", "Y")])
    graph.add_node("Z")
    pos = component_layout(graph)
    assert set(pos) == set(graph.nodes) and len({tuple(round(v, 6) for v in p) for p in pos.values()}) == 6
    assert component_layout(nx.DiGraph()) == {}
