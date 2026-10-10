"""Deterministic synthetic automotive-parts transaction generator.

No suitable, licensed public automotive basket dataset could be obtained, so
the project ships a clearly labelled synthetic dataset. Baskets are drawn from
"purchase occasions" (an oil service, a brake job, ...). Each occasion has a
set of parts with an inclusion probability; some baskets combine two
occasions and most receive a little random noise.

The raw file deliberately contains quality problems (duplicates, missing
values, inconsistent names, non-product rows) so that preprocessing has real
work to do. Every pattern in this data is a consequence of the probabilities
below - it is NOT evidence about real customers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SYNTHETIC_N_TRANSACTIONS, SYNTHETIC_SEED

CATALOGUE: dict[str, str] = {
    "Engine Oil": "Lubricants & Fluids",
    "Transmission Fluid": "Lubricants & Fluids",
    "Gear Oil": "Lubricants & Fluids",
    "Power Steering Fluid": "Lubricants & Fluids",
    "Brake Fluid": "Lubricants & Fluids",
    "Coolant": "Lubricants & Fluids",
    "Windshield Washer Fluid": "Lubricants & Fluids",
    "Engine Flush": "Lubricants & Fluids",
    "Fuel Injector Cleaner": "Lubricants & Fluids",
    "Oil Filter": "Filters",
    "Air Filter": "Filters",
    "Cabin Air Filter": "Filters",
    "Drain Plug Washer": "Filters",
    "Brake Pads": "Brakes",
    "Brake Discs": "Brakes",
    "Brake Cleaner": "Brakes",
    "Spark Plugs": "Ignition & Electrical",
    "Ignition Coil": "Ignition & Electrical",
    "Car Battery": "Ignition & Electrical",
    "Battery Terminals": "Ignition & Electrical",
    "Jump Starter Cables": "Ignition & Electrical",
    "Fuse Set": "Ignition & Electrical",
    "Headlight Bulb": "Lighting & Visibility",
    "Tail Light Bulb": "Lighting & Visibility",
    "Wiper Blades": "Lighting & Visibility",
    "Radiator Hose": "Cooling",
    "Thermostat": "Cooling",
    "Radiator Cap": "Cooling",
    "Shock Absorbers": "Suspension",
    "Suspension Bushings": "Suspension",
    "Wheel Bearing": "Suspension",
    "Tyre Inflator": "Tyres & Wheels",
    "Tyre Pressure Gauge": "Tyres & Wheels",
    "Puncture Repair Kit": "Tyres & Wheels",
    "Valve Caps": "Tyres & Wheels",
    "Car Shampoo": "Car Care",
    "Microfiber Cloth": "Car Care",
    "Car Wax": "Car Care",
    "Tyre Shine": "Car Care",
    "Dashboard Polish": "Car Care",
    "Air Freshener": "Car Care",
}

# occasion -> (weight, {item: inclusion probability})
OCCASIONS: dict[str, tuple[float, dict[str, float]]] = {
    "oil_service": (0.22, {"Engine Oil": 0.95, "Oil Filter": 0.85, "Air Filter": 0.45,
                           "Cabin Air Filter": 0.25, "Engine Flush": 0.20, "Drain Plug Washer": 0.30}),
    "brake_job": (0.12, {"Brake Pads": 0.95, "Brake Fluid": 0.60, "Brake Discs": 0.40, "Brake Cleaner": 0.45}),
    "tune_up": (0.08, {"Spark Plugs": 0.90, "Ignition Coil": 0.35, "Air Filter": 0.50, "Fuel Injector Cleaner": 0.40}),
    "battery": (0.07, {"Car Battery": 0.90, "Battery Terminals": 0.50, "Jump Starter Cables": 0.20}),
    "cooling": (0.07, {"Coolant": 0.90, "Radiator Hose": 0.35, "Thermostat": 0.30, "Radiator Cap": 0.25}),
    "visibility": (0.09, {"Wiper Blades": 0.90, "Windshield Washer Fluid": 0.65, "Headlight Bulb": 0.30}),
    "detailing": (0.12, {"Car Shampoo": 0.85, "Microfiber Cloth": 0.70, "Car Wax": 0.50, "Tyre Shine": 0.40,
                         "Dashboard Polish": 0.40, "Air Freshener": 0.35}),
    "tyres": (0.07, {"Puncture Repair Kit": 0.60, "Tyre Inflator": 0.50, "Tyre Pressure Gauge": 0.50, "Valve Caps": 0.40}),
    "suspension": (0.04, {"Shock Absorbers": 0.90, "Suspension Bushings": 0.50, "Wheel Bearing": 0.30}),
    "road_trip": (0.05, {"Coolant": 0.60, "Windshield Washer Fluid": 0.60, "Tyre Pressure Gauge": 0.55,
                         "Air Freshener": 0.50, "Engine Oil": 0.35, "Jump Starter Cables": 0.30}),
    "lighting": (0.04, {"Headlight Bulb": 0.90, "Fuse Set": 0.50, "Tail Light Bulb": 0.45}),
    "transmission": (0.03, {"Transmission Fluid": 0.90, "Gear Oil": 0.30, "Power Steering Fluid": 0.35}),
}

P_SECOND_OCCASION = 0.25
NOISE_RATE = 0.4  # mean number of unrelated items added per basket
NON_PRODUCT_ROWS = ("Labour Charge", "Service Fee", "Discount")
RAW_COLUMNS = ["transaction_id", "item", "category", "quantity", "transaction_date", "cashier_id"]


def _messy(name: str, rng: np.random.Generator) -> str:
    """Return an inconsistently formatted variant of an item name."""
    variants = (name.upper(), name.lower(), f"  {name} ", name.replace(" ", "  "), name.replace(" ", "_"))
    return variants[int(rng.integers(len(variants)))]


def generate_baskets(n_transactions: int = SYNTHETIC_N_TRANSACTIONS, seed: int = SYNTHETIC_SEED) -> list[list[str]]:
    """Generate clean baskets (lists of canonical item names)."""
    if n_transactions < 1:
        raise ValueError("n_transactions must be at least 1")
    rng = np.random.default_rng(seed)
    names = list(OCCASIONS)
    weights = np.array([OCCASIONS[n][0] for n in names], dtype=float)
    weights /= weights.sum()
    items = list(CATALOGUE)
    noise_w = np.ones(len(items))
    for popular in ("Air Freshener", "Microfiber Cloth", "Engine Oil", "Windshield Washer Fluid"):
        noise_w[items.index(popular)] = 4.0
    noise_w /= noise_w.sum()

    baskets: list[list[str]] = []
    for _ in range(n_transactions):
        k = 2 if rng.random() < P_SECOND_OCCASION else 1
        chosen = rng.choice(len(names), size=k, replace=False, p=weights)
        basket: set[str] = set()
        for idx in chosen:
            for item, p in OCCASIONS[names[int(idx)]][1].items():
                if rng.random() < p:
                    basket.add(item)
        for idx in rng.choice(len(items), size=int(rng.poisson(NOISE_RATE)), p=noise_w):
            basket.add(items[int(idx)])
        if not basket:
            basket.add(items[int(rng.choice(len(items), p=noise_w))])
        baskets.append(sorted(basket))
    return baskets


def generate_raw_transactions(n_transactions: int = SYNTHETIC_N_TRANSACTIONS, seed: int = SYNTHETIC_SEED) -> pd.DataFrame:
    """Generate the raw long-format table, including deliberate quality issues."""
    baskets = generate_baskets(n_transactions, seed)
    rng = np.random.default_rng(seed + 1)
    start = np.datetime64("2025-01-01")
    rows: list[dict] = []
    for i, basket in enumerate(baskets, start=1):
        tid = f"T{i:05d}"
        date = str(start + np.timedelta64(int(rng.integers(0, 365)), "D"))
        cashier = f"C{int(rng.integers(1, 9)):02d}"
        for item in basket:
            name = _messy(item, rng) if rng.random() < 0.03 else item
            rows.append({"transaction_id": tid, "item": name, "category": CATALOGUE[item],
                         "quantity": int(rng.integers(1, 4)), "transaction_date": date, "cashier_id": cashier})
        if rng.random() < 0.06:  # irrelevant non-product invoice line
            rows.append({"transaction_id": tid, "item": NON_PRODUCT_ROWS[int(rng.integers(3))], "category": "Service",
                         "quantity": 1, "transaction_date": date, "cashier_id": cashier})
    # A few invoices that contain no product at all (become empty baskets).
    for j in range(1, 13):
        rows.append({"transaction_id": f"T{n_transactions + j:05d}", "item": "Labour Charge", "category": "Service",
                     "quantity": 1, "transaction_date": str(start + np.timedelta64(int(rng.integers(0, 365)), "D")),
                     "cashier_id": f"C{int(rng.integers(1, 9)):02d}"})

    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    n = len(df)
    # Missing item names and missing transaction identifiers.
    miss_item = rng.choice(n, size=max(1, int(n * 0.005)), replace=False)
    df.loc[miss_item, "item"] = None
    miss_tid = rng.choice(n, size=max(1, int(n * 0.003)), replace=False)
    df.loc[miss_tid, "transaction_id"] = None
    # Exact duplicate rows (double-scanned lines).
    dup = df.iloc[rng.choice(n, size=int(n * 0.02), replace=False)]
    df = pd.concat([df, dup], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
