import pytest

from src.apriori_engine import apriori_native
from src.evaluation import load_manual_review, recommendation_examples
from src.recommender import (compare_thresholds, diff_recommendations, explain_rule, normalize_basket, recommend,
                             recommendations_frame, what_if_remove)
from src.rule_generation import empty_rules, generate_rules
from tests.helpers import fs, onehot

ITEMS = list("ABCD")


def rules_at(min_support=0.4, min_confidence=0.0):
    return generate_rules(apriori_native(onehot(), min_support), min_confidence)


def test_recommends_from_matching_antecedents():
    res = recommend(["A"], rules_at(), top_n=None, known_items=ITEMS)
    assert res.status == "ok" and set(res.items) == {"B", "C"}
    for rec in res.recommendations:
        assert rec.best_rule["antecedent"] <= fs("A")
        assert rec.item in rec.best_rule["consequent"]
        assert rec.confidence == pytest.approx(0.75)     # A->B and A->C are both 3/4


def test_never_recommends_selected_items_or_duplicates():
    res = recommend(["A", "B"], rules_at(), top_n=None, known_items=ITEMS)
    assert res.items == ["C"]
    assert len(res.items) == len(set(res.items))


def test_evidence_is_combined_and_best_rule_is_chosen():
    rec = recommend(["A", "B"], rules_at(), top_n=None).recommendations[0]
    # A->C (0.75), B->C (0.75), AB->C (0.667), A->BC (0.5), B->AC (0.5)
    assert rec.n_rules == 5
    assert rec.confidence == pytest.approx(0.75) and len(rec.best_rule["antecedent"]) == 1
    confs = [r["confidence"] for r in rec.supporting_rules]
    assert confs == sorted(confs, reverse=True)
    assert "4 other matching rule(s)" in rec.explanation


def test_only_rule_supported_recommendations():
    res = recommend(["D"], rules_at(), known_items=ITEMS)     # D is infrequent: no rule mentions it
    assert res.status == "no_match" and res.items == [] and res.unknown_items == []


def test_top_n():
    assert len(recommend(["A"], rules_at(), top_n=1).items) == 1
    with pytest.raises(ValueError):
        recommend(["A"], rules_at(), top_n=0)


def test_empty_rules_and_empty_basket_are_distinct_states():
    assert recommend(["A"], empty_rules()).status == "no_rules"
    assert recommend(["A"], rules_at(1.0)).status == "no_rules"
    assert recommend([], rules_at()).status == "empty_basket"
    assert recommend(["A"], rules_at(0.4, 0.8)).status == "no_rules"


def test_unknown_items_are_reported_not_fatal():
    res = recommend(["A", "Flux Capacitor"], rules_at(), known_items=ITEMS)
    assert res.unknown_items == ["Flux Capacitor"] and set(res.items) == {"B", "C"}


def test_basket_normalisation():
    assert normalize_basket([" engine  oil", "ENGINE_OIL", None, "", "Oil Filter"]) == ["Engine Oil", "Oil Filter"]


def test_explanation_uses_actual_metrics():
    rule = rules_at().query("rule == '{A, B} → {C}'").iloc[0]
    text = explain_rule(rule)
    assert "A and B" in text and "purchased C" in text
    assert "66.7%" in text and "40.0%" in text


def test_basket_changes_update_recommendations():
    rules = rules_at()
    assert set(recommend(["A"], rules, None).items) == {"B", "C"}
    assert recommend(["A", "C"], rules, None).items == ["B"]
    assert recommend(["A", "B", "C"], rules, None).status == "no_match"


def test_what_if_removal():
    out = what_if_remove(["A", "B"], "b", rules_at(), top_n=None)
    assert out["original_basket"] == ["A", "B"] and out["modified_basket"] == ["A"]
    assert out["before"].items == ["C"] and set(out["after"].items) == {"B", "C"}
    assert out["diff"]["added"] == ["B"] and out["diff"]["removed"] == []
    with pytest.raises(ValueError):
        what_if_remove(["A"], "Z", rules_at())


def test_what_if_to_empty_basket():
    out = what_if_remove(["A"], "A", rules_at())
    assert out["after"].status == "empty_basket" and out["diff"]["removed"] == sorted(out["before"].items)


def test_threshold_comparison():
    out = compare_thresholds(onehot(), ["A"], {"min_support": 0.4, "min_confidence": 0.5},
                             {"min_support": 0.4, "min_confidence": 0.8}, top_n=None)
    assert out["a"]["n_rules"] == 12 and out["b"]["n_rules"] == 0
    assert out["a"]["n_itemsets"] == out["b"]["n_itemsets"] == 7
    assert set(out["a"]["result"].items) == {"B", "C"} and out["b"]["result"].status == "no_rules"
    assert out["diff"]["removed"] == ["B", "C"] and out["diff"]["added"] == []


def test_diff_detects_changed_evidence():
    rules = rules_at()
    d = diff_recommendations(recommend(["A", "B"], rules, None), recommend(["A", "B"], rules[rules["confidence"] < 0.7], None))
    assert d["changed"] and d["changed"][0]["item"] == "C"
    assert d["changed"][0]["confidence_after"] == pytest.approx(2 / 3)


def test_frames_and_examples():
    res = recommend(["A"], rules_at(), None)
    frame = recommendations_frame(res)
    assert list(frame["Rank"]) == [1, 2] and frame["Supporting rules"].min() >= 1
    ex = recommendation_examples(rules_at(), [["A"], ["D"]], ITEMS)
    assert (ex["recommended_item"] == "none").sum() == 1 and len(ex) == 3


def test_manual_review_never_invents_rules(tmp_path):
    path = tmp_path / "review.csv"
    path.write_text("antecedent,consequent,classification,reason\n"
                    "A;B,C,Useful,hand written\n"
                    "D,A,Weak / Uncertain,not a real rule\n", encoding="utf-8")
    out = load_manual_review(path, rules_at())
    assert out["discovered"].tolist() == [True, False]
    assert out.loc[0, "confidence"] == pytest.approx(2 / 3)
    assert out.loc[1, ["support", "confidence"]].isna().all()
    path.write_text("antecedent,consequent,classification,reason\nA,B,Great,x\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_manual_review(path, rules_at())
