"""Tests for data.features."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from data.features import ANNUALIZATION, FEATURE_COLUMNS, compute_features


def test_compute_features_adds_all_expected_columns(synthetic_ohlcv_two_symbols: pd.DataFrame) -> None:
    out = compute_features(synthetic_ohlcv_two_symbols)
    for col in FEATURE_COLUMNS:
        assert col in out.columns, f"missing feature column {col}"
    # No NaN in any feature column after warm-up drop.
    assert not out[FEATURE_COLUMNS].isna().any().any()


def test_compute_features_drops_warmup_per_symbol(synthetic_ohlcv_two_symbols: pd.DataFrame) -> None:
    out = compute_features(synthetic_ohlcv_two_symbols)
    # Largest warm-up is the 63-day rolling vol; per-symbol count should be
    # n - 63 (the first 63 rows lose at least one of log_return / rolling_vol_63).
    n_per_symbol = synthetic_ohlcv_two_symbols.groupby("symbol").size().iloc[0]
    for _symbol, group in out.groupby("symbol"):
        # rolling_vol_63 needs 63 prior log returns + 1 base row for the log_return shift.
        assert len(group) == n_per_symbol - 63


def test_rolling_vol_21_loses_21_rows_when_only_that_window_matters() -> None:
    """Verify the rolling_vol_21 warm-up math in isolation."""
    rng = np.random.default_rng(0)
    n = 80
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    log_ret = np.log(close[1:] / close[:-1])
    rolling = pd.Series(log_ret).rolling(21).std()
    valid = rolling.dropna()
    # 80 base rows -> 79 log returns -> 79 - 20 = 59 valid 21-window stds.
    assert len(valid) == 79 - 20


def test_rolling_vols_are_non_negative(synthetic_ohlcv_two_symbols: pd.DataFrame) -> None:
    out = compute_features(synthetic_ohlcv_two_symbols)
    for col in ("rolling_vol_5", "rolling_vol_21", "rolling_vol_63", "ewma_vol_21", "garman_klass_vol", "atr_14"):
        assert (out[col] >= 0.0).all(), f"{col} has negative values"


def test_rsi_within_bounds(synthetic_ohlcv_two_symbols: pd.DataFrame) -> None:
    out = compute_features(synthetic_ohlcv_two_symbols)
    assert out["rsi_14"].between(0.0, 100.0).all()


def test_rolling_vol_63_matches_std_of_last_window() -> None:
    """For a known sequence, rolling_vol_63 / sqrt(252) ~= std(last 63 log returns)."""
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.bdate_range("2019-01-02", periods=n)
    log_ret = rng.normal(0.0, 0.012, size=n)
    log_ret[0] = 0.0
    close = 100.0 * np.exp(np.cumsum(log_ret))
    df = pd.DataFrame(
        {
            "date": dates,
            "symbol": "Z",
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.full(n, 1_000_000, dtype=np.int64),
            "adj_close": close,
        }
    )
    feat = compute_features(df)
    # Reproduce the same rolling std on the recomputed log returns.
    lr = np.log(close[1:] / close[:-1])
    expected_last = float(pd.Series(lr).tail(63).std())
    actual_last = float(feat["rolling_vol_63"].iloc[-1]) / ANNUALIZATION
    assert math.isclose(actual_last, expected_last, rel_tol=1e-6, abs_tol=1e-9)


def test_compute_features_handles_empty_input() -> None:
    empty = pd.DataFrame(
        columns=[
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adj_close",
        ]
    )
    out = compute_features(empty)
    for col in FEATURE_COLUMNS:
        assert col in out.columns
    assert out.empty
