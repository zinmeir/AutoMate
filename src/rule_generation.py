"""Association-rule generation from frequent itemsets.

For a frequent itemset I and a non-empty proper subset A (with B = I \\ A):

    support(A -> B)    = support(A u B) = count(A u B) / N
    confidence(A -> B) = support(A u B) / support(A)

A -> B and B -> A share the same support (same numerator) but are divided by
different antecedent supports, so their confidence generally differs.
Confidence is an observed conditional frequency, not a causal probability.
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd

from .apriori_engine import format_itemset, validate_threshold

RULE_COLUMNS = ["antecedent", "consequent", "support", "confidence", "count",
                "antecedent_support", "consequent_support", "lift", "rule"]


def empty_rules() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=object) for c in RULE_COLUMNS})
    return df.astype({"support": float, "confidence": float, "antecedent_support": float,
                      "consequent_support": float, "lift": float, "count": int})


def format_rule(antecedent, consequent) -> str:
    return f"{format_itemset(antecedent)} → {format_itemset(consequent)}"


def generate_rules(itemsets: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    """Generate every rule A -> B with confidence >= ``min_confidence``.

    Only the itemsets passed in (i.e. those that met the current minimum
    support) are used. ``lift`` is included as a supplementary column; the
    application's thresholds and ranking use support and confidence only.
    """
    min_confidence = validate_threshold(min_confidence, "min_confidence", allow_zero=True)
    if itemsets is None or itemsets.empty:
        return empty_rules()
    support = dict(zip(itemsets["itemset"], itemsets["support"]))
    count = dict(zip(itemsets["itemset"], itemsets["count"]))
    rows = []
    for itemset, sup in support.items():
        if len(itemset) < 2:
            continue
        members = sorted(itemset)
        for r in range(1, len(members)):
            for ante in combinations(members, r):
                a = frozenset(ante)
                b = itemset - a
                # Apriori property: every subset of a frequent itemset is frequent,
                # so both supports are guaranteed to be present.
                conf = sup / support[a]
                if conf >= min_confidence - 1e-12:
                    rows.append((a, b, sup, min(conf, 1.0), int(count[itemset]), support[a], support[b],
                                 conf / support[b], format_rule(a, b)))
    if not rows:
        return empty_rules()
    rules = pd.DataFrame(rows, columns=RULE_COLUMNS)
    return rules.sort_values(["confidence", "support", "rule"], ascending=[False, False, True],
                             kind="mergesort").reset_index(drop=True)


def search_rules(rules: pd.DataFrame, items=None, text: str = "", where: str = "either",
                 match_all: bool = False) -> pd.DataFrame:
    """Filter rules by items and/or free text.

    ``where``: ``antecedent``, ``consequent`` or ``either``.
    ``match_all``: require every selected item (default: any of them).
    ``text``: case-insensitive substring matched against item names.
    """
    if where not in ("antecedent", "consequent", "either"):
        raise ValueError("where must be 'antecedent', 'consequent' or 'either'.")
    if rules.empty:
        return rules
    wanted = {str(i) for i in (items or [])}
    needle = (text or "").strip().lower()

    def side(row) -> frozenset:
        if where == "antecedent":
            return row["antecedent"]
        if where == "consequent":
            return row["consequent"]
        return row["antecedent"] | row["consequent"]

    def keep(row) -> bool:
        s = side(row)
        if wanted and not (wanted <= s if match_all else bool(wanted & s)):
            return False
        return not needle or any(needle in i.lower() for i in s)

    return rules[rules.apply(keep, axis=1)].reset_index(drop=True)


def rules_for_display(rules: pd.DataFrame) -> pd.DataFrame:
    """String-typed copy suitable for tables and CSV export."""
    if rules.empty:
        return pd.DataFrame(columns=["Antecedent (LHS)", "Consequent (RHS)", "Support", "Confidence", "Transactions", "Lift"])
    return pd.DataFrame({
        "Antecedent (LHS)": rules["antecedent"].map(format_itemset),
        "Consequent (RHS)": rules["consequent"].map(format_itemset),
        "Support": rules["support"].astype(float),
        "Confidence": rules["confidence"].astype(float),
        "Transactions": rules["count"].astype(int),
        "Lift": rules["lift"].astype(float),
    })
