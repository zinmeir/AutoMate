"""Tiny hand-verifiable fixtures shared by the tests.

TINY (5 transactions):
    T1 {A, B, C}   T2 {A, B}   T3 {A, C}   T4 {B, C}   T5 {A, B, C, D}

    count: A=4 B=4 C=4 D=1 | AB=3 AC=3 BC=3 | ABC=2
"""
from __future__ import annotations

import pandas as pd

from src.preprocessing import build_baskets, encode_baskets

TINY_BASKETS = {"T1": "ABC", "T2": "AB", "T3": "AC", "T4": "BC", "T5": "ABCD"}
# AB and AC are frequent at 0.4 but BC is not -> {A,B,C} must be pruned without counting.
PRUNE_BASKETS = {"T1": "AB", "T2": "AB", "T3": "AC", "T4": "AC", "T5": "BC"}


def long_frame(baskets: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame([(t, i) for t, items in baskets.items() for i in items], columns=["transaction_id", "item"])


def onehot(baskets: dict[str, str] = TINY_BASKETS) -> pd.DataFrame:
    return encode_baskets(build_baskets(long_frame(baskets)))


def fs(text: str) -> frozenset:
    return frozenset(text)
