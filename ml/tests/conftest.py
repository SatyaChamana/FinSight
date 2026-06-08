"""Shared fixtures: synthetic OHLCV that mimics download_ohlcv output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _build_symbol_ohlcv(symbol: str, n_days: int, start: str, seed: int) -> pd.DataFrame:
    """Generate a plausible OHLCV path with bounded volatility."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    # Daily log returns ~ N(0.0003, 0.012) keeps annualized vol around 19%.
    log_ret = rng.normal(loc=0.0003, scale=0.012, size=n_days)
    log_ret[0] = 0.0
    close = 100.0 * np.exp(np.cumsum(log_ret))
    # Intraday range derived from the daily move so high/low bracket close.
    intra = rng.uniform(0.002, 0.015, size=n_days) * close
    open_ = close * (1.0 + rng.normal(0.0, 0.003, size=n_days))
    high = np.maximum(open_, close) + intra * 0.5
    low = np.minimum(open_, close) - intra * 0.5
    volume = rng.integers(1_000_000, 5_000_000, size=n_days).astype("int64")
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "adj_close": close,
        }
    )


@pytest.fixture
def synthetic_ohlcv_two_symbols() -> pd.DataFrame:
    a = _build_symbol_ohlcv("AAA", n_days=120, start="2020-01-02", seed=1)
    b = _build_symbol_ohlcv("BBB", n_days=120, start="2020-01-02", seed=2)
    return pd.concat([a, b], ignore_index=True)


@pytest.fixture
def synthetic_ohlcv_three_symbols() -> pd.DataFrame:
    a = _build_symbol_ohlcv("AAA", n_days=400, start="2019-01-02", seed=11)
    b = _build_symbol_ohlcv("BBB", n_days=400, start="2019-01-02", seed=22)
    c = _build_symbol_ohlcv("CCC", n_days=400, start="2019-01-02", seed=33)
    return pd.concat([a, b, c], ignore_index=True)


@pytest.fixture
def synthetic_ohlcv_long() -> pd.DataFrame:
    """Ten years of synthetic data for the walk-forward split tests."""
    a = _build_symbol_ohlcv("AAA", n_days=2520, start="2010-01-04", seed=101)
    b = _build_symbol_ohlcv("BBB", n_days=2520, start="2010-01-04", seed=202)
    return pd.concat([a, b], ignore_index=True)
