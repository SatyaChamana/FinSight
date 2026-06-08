"""Tests for data.split.walk_forward_split."""

from __future__ import annotations

import pandas as pd
import pytest

from data.split import walk_forward_split


def test_walk_forward_returns_requested_number_of_splits(
    synthetic_ohlcv_long: pd.DataFrame,
) -> None:
    # Reduce to a single symbol for split testing (split operates on dates).
    df = synthetic_ohlcv_long[synthetic_ohlcv_long["symbol"] == "AAA"].copy()
    splits = walk_forward_split(df, n_splits=3, train_years=4, val_months=6, test_months=6)
    assert len(splits) == 3


def test_walk_forward_train_grows_and_blocks_are_non_overlapping(
    synthetic_ohlcv_long: pd.DataFrame,
) -> None:
    df = synthetic_ohlcv_long[synthetic_ohlcv_long["symbol"] == "AAA"].copy().reset_index(drop=True)
    splits = walk_forward_split(df, n_splits=3, train_years=4, val_months=6, test_months=6)

    dates = pd.to_datetime(df["date"])

    prev_train_size = -1
    for train_idx, val_idx, test_idx in splits:
        # Sets disjoint within a single split.
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(set(val_idx) & set(test_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0

        # Forward in time: max(train) <= min(val) <= max(val) <= min(test).
        if len(train_idx) and len(val_idx):
            assert dates.loc[train_idx].max() <= dates.loc[val_idx].min()
        if len(val_idx) and len(test_idx):
            assert dates.loc[val_idx].max() <= dates.loc[test_idx].min()

        # Train window grows monotonically across splits.
        assert len(train_idx) > prev_train_size
        prev_train_size = len(train_idx)


def test_walk_forward_rejects_zero_splits(synthetic_ohlcv_long: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        walk_forward_split(synthetic_ohlcv_long, n_splits=0)


def test_walk_forward_accepts_date_indexed_frame(synthetic_ohlcv_long: pd.DataFrame) -> None:
    df = (
        synthetic_ohlcv_long[synthetic_ohlcv_long["symbol"] == "AAA"]
        .set_index("date")
        .sort_index()
    )
    splits = walk_forward_split(df, n_splits=2, train_years=4, val_months=6, test_months=6)
    assert len(splits) == 2
    for train_idx, _val_idx, _test_idx in splits:
        # Indices come from the original DatetimeIndex.
        assert isinstance(train_idx, pd.Index)
        if len(train_idx):
            assert isinstance(train_idx[0], pd.Timestamp)
