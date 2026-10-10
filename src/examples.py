"""Illustrative material that is NOT derived from the project dataset."""
from __future__ import annotations

import pandas as pd

from .preprocessing import build_baskets, encode_baskets

# Eight invented baskets used only to walk through Apriori by hand.
TOY_TRANSACTIONS: dict[str, list[str]] = {
    "T1": ["Engine Oil", "Oil Filter", "Air Filter"],
    "T2": ["Engine Oil", "Oil Filter"],
    "T3": ["Engine Oil", "Oil Filter", "Air Filter"],
    "T4": ["Air Filter", "Spark Plugs"],
    "T5": ["Brake Pads", "Brake Fluid"],
    "T6": ["Brake Pads", "Brake Fluid", "Engine Oil"],
    "T7": ["Air Filter", "Spark Plugs", "Coolant"],
    "T8": ["Engine Oil", "Air Filter"],
}
TOY_MIN_SUPPORT = 0.25  # 2 of 8 transactions


def toy_onehot() -> pd.DataFrame:
    frame = pd.DataFrame([(t, i) for t, items in TOY_TRANSACTIONS.items() for i in items],
                         columns=["transaction_id", "item"])
    return encode_baskets(build_baskets(frame))


# Baskets used for the documented recommendation examples (report section 9).
EXAMPLE_BASKETS: list[list[str]] = [
    ["Engine Oil", "Oil Filter"],
    ["Brake Pads"],
    ["Car Shampoo", "Car Wax"],
    ["Wiper Blades"],
    ["Spark Plugs"],
    ["Car Battery"],
    ["Brake Discs", "Brake Fluid"],
]
