"""Evaluation helpers: threshold experiments, examples and the manual rule review."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .apriori_engine import format_itemset, mine_frequent_itemsets
from .recommender import recommend
from .rule_generation import format_rule, generate_rules

CATEGORIES = ("Useful", "Obvious / Trivial", "Unexpected", "Weak / Uncertain")


def item_coverage(rules: pd.DataFrame, items) -> float:
    """Share of items for which a one-item basket receives at least one recommendation."""
    items = list(items)
    if not items or rules.empty:
        return 0.0
    singles = {next(iter(a)) for a in rules["antecedent"] if len(a) == 1}
    return len(singles & set(items)) / len(items)


def threshold_experiment(onehot: pd.DataFrame, supports, confidences, basket=None, top_n: int = 3,
                         backend: str = "native") -> pd.DataFrame:
    """Mine the same data under every (min_support, min_confidence) combination."""
    rows = []
    for s in supports:
        itemsets = mine_frequent_itemsets(onehot, s, backend=backend)
        for c in confidences:
            rules = generate_rules(itemsets, c)
            row = {"min_support": s, "min_confidence": c, "transactions": len(onehot),
                   "frequent_itemsets": len(itemsets),
                   "largest_itemset": int(itemsets["size"].max()) if len(itemsets) else 0,
                   "rules": len(rules),
                   "items_with_recommendation": item_coverage(rules, onehot.columns)}
            if basket:
                res = recommend(basket, rules, top_n, known_items=list(onehot.columns))
                row["example_basket"] = format_itemset(res.basket)
                row["example_recommendations"] = ", ".join(
                    f"{r.item} ({r.confidence:.0%})" for r in res.recommendations) or "none"
            rows.append(row)
    return pd.DataFrame(rows)


def recommendation_examples(rules: pd.DataFrame, baskets, known_items, top_n: int = 3) -> pd.DataFrame:
    """One row per (basket, recommended item) with its best supporting rule."""
    rows = []
    for basket in baskets:
        res = recommend(basket, rules, top_n, known_items=known_items)
        if not res.recommendations:
            rows.append({"basket": format_itemset(res.basket), "recommended_item": "none", "supporting_rule": "-",
                         "support": None, "confidence": None, "supporting_rules": 0, "explanation": res.message})
        for r in res.recommendations:
            rows.append({"basket": format_itemset(res.basket), "recommended_item": r.item,
                         "supporting_rule": r.best_rule["rule"], "support": r.support, "confidence": r.confidence,
                         "supporting_rules": r.n_rules, "explanation": r.explanation})
    return pd.DataFrame(rows)


def load_manual_review(path: str | Path, rules: pd.DataFrame) -> pd.DataFrame:
    """Join the hand-written rule classifications with the rules actually mined.

    The review file stores ``antecedent``/``consequent`` as ``;``-separated item
    names plus a human-assigned ``classification`` and ``reason``. Metrics are
    never stored in the file: they are always looked up from ``rules``, and
    ``discovered`` is False for any reviewed rule the current mining run did
    not produce (so the table can never contain a rule that does not exist).
    """
    review = pd.read_csv(path)
    needed = {"antecedent", "consequent", "classification", "reason"}
    if not needed <= set(review.columns):
        raise ValueError(f"Manual review file must contain columns: {sorted(needed)}")
    bad = set(review["classification"]) - set(CATEGORIES)
    if bad:
        raise ValueError(f"Unknown classification(s) {sorted(bad)}; allowed: {CATEGORIES}")
    lookup = {r["rule"]: r for r in rules.to_dict("records")} if not rules.empty else {}
    rows = []
    for rec in review.to_dict("records"):
        a = frozenset(p.strip() for p in str(rec["antecedent"]).split(";") if p.strip())
        b = frozenset(p.strip() for p in str(rec["consequent"]).split(";") if p.strip())
        label = format_rule(a, b)
        hit = lookup.get(label)
        rows.append({"rule": label, "support": hit["support"] if hit else None,
                     "confidence": hit["confidence"] if hit else None, "lift": hit["lift"] if hit else None,
                     "classification": rec["classification"], "reason": rec["reason"], "discovered": hit is not None})
    return pd.DataFrame(rows)
