"""Frequent-itemset mining with the Apriori algorithm.

``apriori_native`` is a from-scratch, level-wise implementation that can also
return a step-by-step trace (candidates generated, candidates pruned, counts).
``mine_frequent_itemsets`` is the entry point used by the application; it can
alternatively delegate to ``mlxtend`` when that library is installed.

Support is always a fraction in [0, 1]:

    support(X) = (number of transactions containing every item of X) / (number of transactions)
"""
from __future__ import annotations

import math
from itertools import combinations
from numbers import Real

import numpy as np
import pandas as pd

from .preprocessing import validate_onehot

ITEMSET_COLUMNS = ["itemset", "size", "support", "count"]
_EPS = 1e-12


def validate_threshold(value: object, name: str, allow_zero: bool = False) -> float:
    """Validate a support/confidence threshold and return it as ``float``."""
    if isinstance(value, bool) or not isinstance(value, Real) or math.isnan(float(value)):
        raise ValueError(f"{name} must be a number between 0 and 1, got {value!r}.")
    value = float(value)
    low_ok = value >= 0 if allow_zero else value > 0
    if not (low_ok and value <= 1):
        bound = "0 <= value <= 1" if allow_zero else "0 < value <= 1"
        raise ValueError(f"{name} must satisfy {bound}, got {value}.")
    return value


def format_itemset(items) -> str:
    """Human-readable, deterministic label such as ``{Engine Oil, Oil Filter}``."""
    return "{" + ", ".join(sorted(items)) + "}"


def empty_itemsets() -> pd.DataFrame:
    return pd.DataFrame({"itemset": pd.Series(dtype=object), "size": pd.Series(dtype=int),
                         "support": pd.Series(dtype=float), "count": pd.Series(dtype=int)})


def generate_candidates(prev_frequent: list[tuple[str, ...]]) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """Apriori-gen: join step followed by the subset-pruning step.

    ``prev_frequent`` holds the frequent (k-1)-itemsets as sorted tuples.
    Returns ``(kept, pruned)`` where ``pruned`` are joined candidates that were
    discarded because at least one of their (k-1)-subsets is not frequent
    (the Apriori property), without ever being counted against the data.
    """
    prev = sorted(prev_frequent)
    prev_set = set(prev)
    kept, pruned = [], []
    for i, a in enumerate(prev):
        for b in prev[i + 1:]:
            if a[:-1] != b[:-1]:
                break  # sorted order: no later b shares the prefix either
            cand = a + (b[-1],)
            if all(sub in prev_set for sub in combinations(cand, len(cand) - 1)):
                kept.append(cand)
            else:
                pruned.append(cand)
    return kept, pruned


def apriori_native(onehot: pd.DataFrame, min_support: float, max_len: int | None = None,
                   return_trace: bool = False):
    """Level-wise Apriori on a boolean one-hot matrix.

    Returns a DataFrame with columns ``itemset`` (frozenset), ``size``,
    ``support`` and ``count``. With ``return_trace=True`` a second value is
    returned: one dict per level describing candidate generation and pruning.
    """
    min_support = validate_threshold(min_support, "min_support")
    validate_onehot(onehot)
    if max_len is not None and max_len < 1:
        raise ValueError("max_len must be at least 1.")
    n = len(onehot)
    cols = [str(c) for c in onehot.columns]
    matrix = onehot.to_numpy(dtype=bool)
    column = {c: matrix[:, j] for j, c in enumerate(cols)}

    def is_frequent(count: int) -> bool:
        return count / n >= min_support - _EPS

    rows: list[tuple[frozenset, int, float, int]] = []
    trace: list[dict] = []

    # Level 1: every single item is a candidate.
    counts1 = {(c,): int(column[c].sum()) for c in sorted(cols)}
    vectors = {k: column[k[0]] for k, v in counts1.items() if is_frequent(v)}
    trace.append({"k": 1, "joined": list(counts1), "pruned_by_subset": [], "counted": dict(counts1),
                  "frequent": sorted(vectors), "pruned_by_support": sorted(k for k in counts1 if k not in vectors)})
    rows += [(frozenset(k), 1, counts1[k] / n, counts1[k]) for k in sorted(vectors)]

    k = 2
    while vectors and (max_len is None or k <= max_len):
        kept, pruned = generate_candidates(list(vectors))
        if not kept and not pruned:
            break
        counted, new_vectors = {}, {}
        for cand in kept:
            vec = vectors[cand[:-1]] & column[cand[-1]]
            cnt = int(vec.sum())
            counted[cand] = cnt
            if is_frequent(cnt):
                new_vectors[cand] = vec
        trace.append({"k": k, "joined": kept + pruned, "pruned_by_subset": pruned, "counted": counted,
                      "frequent": sorted(new_vectors),
                      "pruned_by_support": sorted(c for c in counted if c not in new_vectors)})
        rows += [(frozenset(c), k, counted[c] / n, counted[c]) for c in sorted(new_vectors)]
        vectors = new_vectors
        k += 1

    result = _finalise(pd.DataFrame(rows, columns=ITEMSET_COLUMNS)) if rows else empty_itemsets()
    return (result, trace) if return_trace else result


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(_label=df["itemset"].map(format_itemset))
    df = df.sort_values(["size", "support", "_label"], ascending=[True, False, True], kind="mergesort")
    return df.drop(columns="_label").reset_index(drop=True).astype({"size": int, "count": int, "support": float})


def mlxtend_available() -> bool:
    try:
        import mlxtend.frequent_patterns  # noqa: F401
        return True
    except Exception:
        return False


def apriori_mlxtend(onehot: pd.DataFrame, min_support: float) -> pd.DataFrame:
    """Same output format as ``apriori_native`` but mined by ``mlxtend``."""
    min_support = validate_threshold(min_support, "min_support")
    validate_onehot(onehot)
    from mlxtend.frequent_patterns import apriori

    found = apriori(onehot.astype(bool), min_support=min_support, use_colnames=True)
    if found.empty:
        return empty_itemsets()
    n = len(onehot)
    out = pd.DataFrame({"itemset": [frozenset(map(str, s)) for s in found["itemsets"]],
                        "support": found["support"].astype(float).to_numpy()})
    out["size"] = out["itemset"].map(len)
    out["count"] = (out["support"] * n).round().astype(int)
    out["support"] = out["count"] / n  # remove floating-point noise
    return _finalise(out[ITEMSET_COLUMNS])


def mine_frequent_itemsets(onehot: pd.DataFrame, min_support: float, backend: str = "native") -> pd.DataFrame:
    """Mine frequent itemsets with the chosen backend (``native`` or ``mlxtend``)."""
    if backend == "native":
        return apriori_native(onehot, min_support)
    if backend == "mlxtend":
        return apriori_mlxtend(onehot, min_support)
    raise ValueError(f"Unknown backend {backend!r}; use 'native' or 'mlxtend'.")


def trace_to_frame(trace: list[dict], n_transactions: int, min_support: float) -> pd.DataFrame:
    """Flatten an Apriori trace into one row per candidate (for display)."""
    rows = []
    for level in trace:
        frequent = set(level["frequent"])
        for cand in level["joined"]:
            if cand in level["pruned_by_subset"]:
                rows.append({"k": level["k"], "candidate": format_itemset(cand), "count": None, "support": None,
                             "outcome": "Pruned before counting (has an infrequent subset)"})
            else:
                cnt = level["counted"][cand]
                rows.append({"k": level["k"], "candidate": format_itemset(cand), "count": cnt,
                             "support": cnt / n_transactions,
                             "outcome": "Frequent" if cand in frequent else f"Pruned (support < {min_support:g})"})
    frame = pd.DataFrame(rows, columns=["k", "candidate", "count", "support", "outcome"])
    return frame.astype({"count": "Int64", "support": "float64"})
