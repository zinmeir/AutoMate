"""Cleaning raw rows and converting them to baskets / a one-hot matrix.

Every step records how many rows it removed or changed, so nothing is
discarded silently.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .config import NON_PRODUCT_ITEMS
from .data_loader import validate_schema

_WS = re.compile(r"\s+")


def normalize_item(name: object) -> str:
    """Canonical item name: trimmed, single-spaced, underscores as spaces, Title Case.

    Returns an empty string for missing/blank values.
    """
    if name is None or (not isinstance(name, str) and pd.isna(name)):
        return ""
    text = _WS.sub(" ", str(name).replace("_", " ")).strip()
    return text.title()


@dataclass
class CleaningReport:
    steps: list[dict] = field(default_factory=list)
    raw_rows: int = 0
    clean_rows: int = 0
    raw_transactions: int = 0
    clean_transactions: int = 0
    empty_transactions_dropped: int = 0

    def add(self, step: str, before: int, after: int, changed: int, reason: str) -> None:
        self.steps.append({"step": step, "rows_before": before, "rows_after": after,
                           "rows_removed": before - after, "rows_changed": changed, "reason": reason})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    @property
    def rows_removed_fraction(self) -> float:
        return 0.0 if self.raw_rows == 0 else 1 - self.clean_rows / self.raw_rows


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Return a tidy ``transaction_id, item`` table and an audit report."""
    validate_schema(raw)
    rep = CleaningReport(raw_rows=len(raw))
    df = raw.copy()
    rep.raw_transactions = int(df["transaction_id"].dropna().astype(str).str.strip().replace("", pd.NA).nunique())

    # 1. keep only the fields the mining pipeline needs
    extra = [c for c in df.columns if c not in ("transaction_id", "item")]
    df = df[["transaction_id", "item"]]
    rep.add("Select mining columns", len(df), len(df), 0,
            f"Dropped irrelevant column(s): {', '.join(extra) if extra else 'none'}. No rows affected.")

    # 2. missing transaction identifiers
    before = len(df)
    tid = df["transaction_id"].astype("string").str.strip()
    df = df.assign(transaction_id=tid)[tid.notna() & (tid != "")]
    rep.add("Remove rows with missing transaction_id", before, len(df), 0,
            "A line that cannot be assigned to a basket cannot contribute to any itemset.")

    # 3. missing item names
    before = len(df)
    norm = df["item"].map(normalize_item)
    keep = norm != ""
    df, norm, original = df[keep], norm[keep], df["item"][keep]
    rep.add("Remove rows with missing item", before, len(df), 0, "Blank or null item names carry no information.")

    # 4. normalise names
    changed = int((original.astype(str) != norm).sum())
    df = df.assign(item=norm)
    rep.add("Normalise item names", len(df), len(df), changed,
            "Trimmed whitespace, collapsed repeated spaces, replaced underscores, applied Title Case "
            "so that e.g. 'ENGINE  OIL' and 'engine_oil' count as one item.")

    # 5. irrelevant non-product lines
    before = len(df)
    df = df[~df["item"].str.lower().isin(NON_PRODUCT_ITEMS)]
    rep.add("Remove non-product lines", before, len(df), 0,
            "Invoice lines such as labour charges, fees and discounts are not purchasable parts.")

    # 6. duplicate (transaction, item) pairs
    before = len(df)
    df = df.drop_duplicates(subset=["transaction_id", "item"])
    rep.add("Remove duplicate transaction-item pairs", before, len(df), 0,
            "An item counts once per basket; repeated lines must not inflate support.")

    df = df.sort_values(["transaction_id", "item"]).reset_index(drop=True)
    rep.clean_rows = len(df)
    rep.clean_transactions = int(df["transaction_id"].nunique())
    rep.empty_transactions_dropped = rep.raw_transactions - rep.clean_transactions
    return df, rep


def build_baskets(clean: pd.DataFrame) -> dict[str, frozenset[str]]:
    """Group a tidy table into ``{transaction_id: frozenset(items)}``."""
    if clean.empty:
        return {}
    grouped = clean.groupby("transaction_id", sort=True)["item"].agg(frozenset)
    return {str(k): v for k, v in grouped.items()}


def encode_baskets(baskets: dict[str, frozenset[str]]) -> pd.DataFrame:
    """One-hot encode baskets: rows = transactions, columns = items, values = bool."""
    if not baskets:
        return pd.DataFrame(dtype=bool)
    items = sorted({i for b in baskets.values() for i in b})
    index = {item: j for j, item in enumerate(items)}
    import numpy as np

    matrix = np.zeros((len(baskets), len(items)), dtype=bool)
    for r, basket in enumerate(baskets.values()):
        for item in basket:
            matrix[r, index[item]] = True
    return pd.DataFrame(matrix, index=pd.Index(list(baskets), name="transaction_id"), columns=items)


def validate_onehot(onehot: pd.DataFrame) -> None:
    """Raise ``ValueError`` if the matrix is unusable for mining."""
    if onehot is None or onehot.shape[0] == 0 or onehot.shape[1] == 0:
        raise ValueError("The transaction matrix is empty - there is nothing to mine.")
    if not all(str(dt) == "bool" for dt in onehot.dtypes):
        raise ValueError("The transaction matrix must be boolean (one-hot encoded).")
    if not onehot.to_numpy().any(axis=1).all():
        raise ValueError("The transaction matrix contains empty transactions.")


def summary_stats(onehot: pd.DataFrame) -> dict:
    """Dataset summary used by the overview page and the report."""
    sizes = onehot.sum(axis=1)
    n_tx, n_items = onehot.shape
    return {
        "transactions": int(n_tx),
        "unique_items": int(n_items),
        "item_occurrences": int(sizes.sum()),
        "avg_basket_size": float(sizes.mean()) if n_tx else 0.0,
        "median_basket_size": float(sizes.median()) if n_tx else 0.0,
        "min_basket_size": int(sizes.min()) if n_tx else 0,
        "max_basket_size": int(sizes.max()) if n_tx else 0,
        "single_item_baskets": int((sizes == 1).sum()),
        "density": float(onehot.to_numpy().mean()) if n_tx and n_items else 0.0,
    }


def item_frequencies(onehot: pd.DataFrame) -> pd.DataFrame:
    """Per-item transaction count and support, most frequent first."""
    counts = onehot.sum(axis=0).astype(int)
    out = pd.DataFrame({"item": counts.index, "count": counts.to_numpy(),
                        "support": counts.to_numpy() / max(len(onehot), 1)})
    return out.sort_values(["count", "item"], ascending=[False, True]).reset_index(drop=True)
