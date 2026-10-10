"""Generate the synthetic raw dataset and the cleaned dataset.

    python scripts/prepare_dataset.py

Writes data/raw_transactions.csv (with deliberate quality issues) and
data/transactions.csv (cleaned, long format). Deterministic for a fixed seed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CLEAN_DATA_PATH, DATA_DIR, RAW_DATA_PATH, SYNTHETIC_N_TRANSACTIONS, SYNTHETIC_SEED  # noqa: E402
from src.preprocessing import clean_transactions  # noqa: E402
from src.synthetic_data import generate_raw_transactions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=int, default=SYNTHETIC_N_TRANSACTIONS)
    parser.add_argument("--seed", type=int, default=SYNTHETIC_SEED)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = generate_raw_transactions(args.transactions, args.seed)
    raw.to_csv(RAW_DATA_PATH, index=False)
    clean, report = clean_transactions(raw)
    clean.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Raw rows: {report.raw_rows} -> {RAW_DATA_PATH}")
    print(f"Clean rows: {report.clean_rows} in {report.clean_transactions} transactions -> {CLEAN_DATA_PATH}")
    print(report.to_frame()[["step", "rows_before", "rows_after", "rows_changed"]].to_string(index=False))


if __name__ == "__main__":
    main()
