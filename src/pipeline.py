"""End-to-end helpers shared by the app and the scripts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .apriori_engine import mine_frequent_itemsets
from .data_loader import Provenance, load_transactions
from .preprocessing import CleaningReport, build_baskets, clean_transactions, encode_baskets, validate_onehot
from .rule_generation import generate_rules


@dataclass
class PreparedData:
    raw: pd.DataFrame
    clean: pd.DataFrame
    onehot: pd.DataFrame
    report: CleaningReport
    provenance: Provenance


def prepare_data(path: str | Path | None = None) -> PreparedData:
    """Dataset -> preprocessing -> transactions -> one-hot matrix."""
    raw, provenance = load_transactions(path)
    clean, report = clean_transactions(raw)
    onehot = encode_baskets(build_baskets(clean))
    validate_onehot(onehot)
    return PreparedData(raw=raw, clean=clean, onehot=onehot, report=report, provenance=provenance)


def mine(onehot: pd.DataFrame, min_support: float, min_confidence: float, backend: str = "native"):
    """Apriori -> frequent itemsets -> rules. Returns ``(itemsets, rules)``."""
    itemsets = mine_frequent_itemsets(onehot, min_support, backend=backend)
    return itemsets, generate_rules(itemsets, min_confidence)
