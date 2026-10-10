import pandas as pd
import pytest

from src.data_loader import DataValidationError, load_transactions, validate_schema
from src.preprocessing import (build_baskets, clean_transactions, encode_baskets, item_frequencies, normalize_item,
                               summary_stats, validate_onehot)
from src.synthetic_data import generate_raw_transactions
from tests.helpers import TINY_BASKETS, long_frame, onehot

MESSY = pd.DataFrame({
    "transaction_id": ["T1", "T1", "T1", "T1", "T2", "T2", None, "T3", "T4", "T4", " T5 "],
    "item": ["Engine Oil", "ENGINE  OIL", " oil_filter ", "Labour Charge", "Brake Pads", None, "Coolant",
             "Service Fee", "Coolant", "Coolant", "wiper blades"],
    "cashier_id": ["C1"] * 11,
})


def test_normalize_item():
    assert normalize_item("  ENGINE   oil ") == "Engine Oil"
    assert normalize_item("oil_filter") == "Oil Filter"
    assert normalize_item(None) == ""
    assert normalize_item(float("nan")) == ""
    assert normalize_item("   ") == ""


def test_load_valid_file(tmp_path):
    path = tmp_path / "tx.csv"
    long_frame(TINY_BASKETS).to_csv(path, index=False)
    df, prov = load_transactions(path)
    assert len(df) == 13 and not prov.used_fallback
    assert list(df.columns) == ["transaction_id", "item"]


def test_schema_validation():
    with pytest.raises(DataValidationError):
        validate_schema(pd.DataFrame({"transaction_id": ["T1"], "product": ["x"]}))
    with pytest.raises(DataValidationError):
        validate_schema(pd.DataFrame({"transaction_id": [], "item": []}))


def test_fallback_when_file_missing(tmp_path):
    df, prov = load_transactions(tmp_path / "does_not_exist.csv")
    assert prov.used_fallback and prov.is_synthetic and len(df) > 0
    with pytest.raises(DataValidationError):
        load_transactions(tmp_path / "does_not_exist.csv", allow_fallback=False)


def test_fallback_when_schema_is_wrong(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(path, index=False)
    _, prov = load_transactions(path)
    assert prov.used_fallback and "could not be used" in prov.note


def test_cleaning_handles_missing_duplicates_and_irrelevant_rows():
    clean, rep = clean_transactions(MESSY)
    steps = {s["step"]: s for s in rep.steps}
    assert steps["Remove rows with missing transaction_id"]["rows_removed"] == 1
    assert steps["Remove rows with missing item"]["rows_removed"] == 1
    assert steps["Remove non-product lines"]["rows_removed"] == 2
    # ENGINE  OIL duplicates Engine Oil in T1; Coolant is repeated in T4
    assert steps["Remove duplicate transaction-item pairs"]["rows_removed"] == 2
    assert steps["Normalise item names"]["rows_changed"] == 3
    assert list(clean.columns) == ["transaction_id", "item"]
    assert rep.raw_rows == 11 and rep.clean_rows == 5
    # every removed row is accounted for
    assert sum(s["rows_removed"] for s in rep.steps) == rep.raw_rows - rep.clean_rows


def test_empty_transactions_are_dropped_and_counted():
    _, rep = clean_transactions(MESSY)
    assert rep.raw_transactions == 5          # T1..T5
    assert rep.clean_transactions == 4        # T3 contained only a service fee
    assert rep.empty_transactions_dropped == 1


def test_baskets_group_correctly():
    clean, _ = clean_transactions(MESSY)
    baskets = build_baskets(clean)
    assert baskets == {"T1": frozenset({"Engine Oil", "Oil Filter"}), "T2": frozenset({"Brake Pads"}),
                       "T4": frozenset({"Coolant"}), "T5": frozenset({"Wiper Blades"})}


def test_one_hot_encoding():
    oh = onehot()
    assert oh.shape == (5, 4) and list(oh.columns) == list("ABCD")
    assert all(str(d) == "bool" for d in oh.dtypes)
    assert oh.loc["T2"].tolist() == [True, True, False, False]
    assert oh.sum().to_dict() == {"A": 4, "B": 4, "C": 4, "D": 1}
    validate_onehot(oh)


def test_duplicates_do_not_inflate_support():
    doubled = pd.concat([long_frame(TINY_BASKETS)] * 2, ignore_index=True)
    clean, _ = clean_transactions(doubled)
    assert encode_baskets(build_baskets(clean)).sum().to_dict() == {"A": 4, "B": 4, "C": 4, "D": 1}


def test_validate_onehot_rejects_empty():
    with pytest.raises(ValueError):
        validate_onehot(pd.DataFrame())
    with pytest.raises(ValueError):
        validate_onehot(encode_baskets({}))


def test_summary_and_frequencies():
    oh = onehot()
    stats = summary_stats(oh)
    assert stats["transactions"] == 5 and stats["unique_items"] == 4
    assert stats["avg_basket_size"] == pytest.approx(13 / 5)
    freq = item_frequencies(oh)
    assert freq.iloc[0]["count"] == 4 and freq.iloc[-1]["item"] == "D"
    assert freq.iloc[-1]["support"] == pytest.approx(0.2)


def test_synthetic_generator_is_deterministic():
    a = generate_raw_transactions(200, seed=1)
    b = generate_raw_transactions(200, seed=1)
    c = generate_raw_transactions(200, seed=2)
    assert a.equals(b) and not a.equals(c)
