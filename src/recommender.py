"""Rule-based recommendation engine with explanations.

Ranking method (transparent and deterministic - not claimed to be optimal):

1. A rule *matches* a basket when its antecedent is a subset of the basket.
2. Every consequent item that is not already in the basket becomes a candidate.
3. All matching rules for a candidate are kept as evidence. The *best* rule is
   the one with the highest confidence, then highest support, then the larger
   (more specific) antecedent.
4. Candidates are ranked by their best rule using the same key, then by the
   number of supporting rules, then alphabetically.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .apriori_engine import format_itemset, mine_frequent_itemsets
from .preprocessing import normalize_item
from .rule_generation import generate_rules

RANKING_DESCRIPTION = (
    "Ranked by the best supporting rule: highest confidence first, then highest support, "
    "then the more specific (larger) antecedent, then the number of supporting rules."
)


def _pct(x: float) -> str:
    return f"{100 * float(x):.1f}%"


def _join_words(items) -> str:
    items = sorted(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def explain_rule(rule) -> str:
    """Turn one rule (row/dict with antecedent, consequent, support, confidence) into a sentence."""
    a, b = rule["antecedent"], rule["consequent"]
    return (f"Customers whose transactions contained {_join_words(a)} also purchased {_join_words(b)} "
            f"in {_pct(rule['confidence'])} of the transactions containing that antecedent. "
            f"This rule has support {_pct(rule['support'])} across all transactions.")


@dataclass
class Recommendation:
    item: str
    confidence: float           # of the best supporting rule
    support: float              # of the best supporting rule
    best_rule: dict
    supporting_rules: list[dict]
    explanation: str

    @property
    def n_rules(self) -> int:
        return len(self.supporting_rules)


@dataclass
class RecommendationResult:
    basket: list[str]
    recommendations: list[Recommendation] = field(default_factory=list)
    unknown_items: list[str] = field(default_factory=list)
    matched_rules: int = 0
    status: str = "ok"          # ok | empty_basket | no_rules | no_match
    message: str = ""

    @property
    def items(self) -> list[str]:
        return [r.item for r in self.recommendations]


def normalize_basket(basket) -> list[str]:
    """Normalised, de-duplicated basket preserving first-seen order."""
    seen, out = set(), []
    for raw in basket or []:
        item = normalize_item(raw)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _rule_key(rule: dict) -> tuple:
    return (-rule["confidence"], -rule["support"], -len(rule["antecedent"]), rule["rule"])


def recommend(basket, rules: pd.DataFrame, top_n: int | None = 5, known_items=None) -> RecommendationResult:
    """Recommend items for ``basket`` using ``rules``.

    ``known_items``: the dataset's item vocabulary; basket entries outside it
    are reported in ``unknown_items`` (they can never match a rule).
    """
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be at least 1 (or None for all).")
    items = normalize_basket(basket)
    result = RecommendationResult(basket=items)
    if known_items is not None:
        vocab = set(known_items)
        result.unknown_items = [i for i in items if i not in vocab]
    if not items:
        result.status, result.message = "empty_basket", "The basket is empty - add at least one item."
        return result
    if rules is None or rules.empty:
        result.status = "no_rules"
        result.message = ("No association rules satisfy the current thresholds, so nothing can be recommended. "
                          "Lower minimum support and/or minimum confidence.")
        return result

    basket_set = frozenset(items)
    matching = rules[rules["antecedent"].map(lambda a: a <= basket_set)]
    result.matched_rules = int(len(matching))
    evidence: dict[str, list[dict]] = {}
    for rule in matching.to_dict("records"):
        for item in rule["consequent"] - basket_set:
            evidence.setdefault(item, []).append(rule)
    if not evidence:
        result.status = "no_match"
        result.message = ("Rules exist at these thresholds, but none has an antecedent contained in this basket "
                          "with a consequent outside it. This is a normal outcome, not an error.")
        return result

    recs = []
    for item, rule_list in evidence.items():
        rule_list.sort(key=_rule_key)
        best = rule_list[0]
        text = explain_rule(best)
        if len(rule_list) > 1:
            text += f" {len(rule_list) - 1} other matching rule(s) also point to {item}."
        recs.append(Recommendation(item=item, confidence=float(best["confidence"]), support=float(best["support"]),
                                   best_rule=best, supporting_rules=rule_list, explanation=text))
    recs.sort(key=lambda r: (_rule_key(r.best_rule)[:3], -r.n_rules, r.item))
    result.recommendations = recs if top_n is None else recs[:top_n]
    return result


def recommendations_frame(result: RecommendationResult) -> pd.DataFrame:
    """Tabular view of a recommendation result."""
    return pd.DataFrame([{
        "Rank": i, "Recommended item": r.item,
        "Best rule": r.best_rule["rule"], "Confidence": r.confidence, "Support": r.support,
        "Supporting rules": r.n_rules,
    } for i, r in enumerate(result.recommendations, start=1)],
        columns=["Rank", "Recommended item", "Best rule", "Confidence", "Support", "Supporting rules"])


def diff_recommendations(before: RecommendationResult, after: RecommendationResult) -> dict:
    """Compare two results: which items appeared, disappeared, or changed evidence/rank."""
    b = {r.item: (i, r) for i, r in enumerate(before.recommendations, start=1)}
    a = {r.item: (i, r) for i, r in enumerate(after.recommendations, start=1)}
    changed = []
    for item in sorted(set(a) & set(b)):
        (rb, xb), (ra, xa) = b[item], a[item]
        if rb != ra or xb.best_rule["rule"] != xa.best_rule["rule"] or abs(xb.confidence - xa.confidence) > 1e-12:
            changed.append({"item": item, "rank_before": rb, "rank_after": ra,
                            "confidence_before": xb.confidence, "confidence_after": xa.confidence,
                            "rule_before": xb.best_rule["rule"], "rule_after": xa.best_rule["rule"]})
    return {"added": sorted(set(a) - set(b)), "removed": sorted(set(b) - set(a)),
            "unchanged": sorted(i for i in set(a) & set(b) if i not in {c["item"] for c in changed}),
            "changed": changed}


def what_if_remove(basket, item_to_remove, rules: pd.DataFrame, top_n: int | None = 5, known_items=None) -> dict:
    """Recommendations before and after removing one item from the basket.

    This compares rule outputs for two baskets; it is not a causal prediction.
    """
    original = normalize_basket(basket)
    target = normalize_item(item_to_remove)
    if target not in original:
        raise ValueError(f"{target!r} is not in the basket.")
    modified = [i for i in original if i != target]
    before = recommend(original, rules, top_n, known_items)
    after = recommend(modified, rules, top_n, known_items)
    return {"original_basket": original, "modified_basket": modified, "removed_item": target,
            "before": before, "after": after, "diff": diff_recommendations(before, after)}


def compare_thresholds(onehot: pd.DataFrame, basket, config_a: dict, config_b: dict, top_n: int | None = 5,
                       backend: str = "native") -> dict:
    """Mine twice on the same data and compare recommendations for the same basket.

    Each config is ``{"min_support": float, "min_confidence": float}``.
    """
    out = {}
    for label, cfg in (("a", config_a), ("b", config_b)):
        itemsets = mine_frequent_itemsets(onehot, cfg["min_support"], backend=backend)
        rules = generate_rules(itemsets, cfg["min_confidence"])
        out[label] = {"config": dict(cfg), "n_itemsets": int(len(itemsets)), "n_rules": int(len(rules)),
                      "result": recommend(basket, rules, top_n, known_items=list(onehot.columns))}
    out["diff"] = diff_recommendations(out["a"]["result"], out["b"]["result"])
    return out
