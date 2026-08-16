import pandas as pd
import pytest

from ml.config import Config
from ml.data.loader import RAW_COLUMNS, load_raw


@pytest.fixture(scope="module")
def df():
    return load_raw(Config.load().get("data.csv_path"))


def test_row_and_column_count(df):
    assert df.shape == (2512, 16)


def test_all_raw_columns_present(df):
    assert list(df.columns) == RAW_COLUMNS


def test_dates_parsed_to_datetime(df):
    assert pd.api.types.is_datetime64_any_dtype(df["TransactionDate"])
    assert pd.api.types.is_datetime64_any_dtype(df["PreviousTransactionDate"])


def test_no_missing_values(df):
    assert df.isnull().sum().sum() == 0


def test_index_is_a_clean_range(df):
    assert df.index.tolist() == list(range(2512))


def test_missing_column_is_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("TransactionID,AccountID\nTX1,AC1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_raw(bad)
