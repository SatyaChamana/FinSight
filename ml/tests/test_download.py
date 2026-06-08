"""Tests for data.download.

We avoid hitting the network: instead we pre-write a parquet cache file and
assert the cache-hit path returns it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.download import OHLCV_COLUMNS, download_ohlcv


def _write_cache_file(cache_dir: Path, symbol: str) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "symbol": [symbol] * 3,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.5, 100.5, 101.5],
            "close": [100.5, 101.5, 102.5],
            "volume": [1_000_000, 1_100_000, 1_050_000],
            "adj_close": [100.5, 101.5, 102.5],
        }
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_dir / f"{symbol}.parquet", index=False)
    return df


def test_download_returns_empty_for_empty_symbol_list(tmp_path: Path) -> None:
    out = download_ohlcv([], start="2024-01-01", end="2024-02-01", cache_dir=tmp_path)
    assert list(out.columns) == OHLCV_COLUMNS
    assert out.empty


def test_download_uses_cache_when_present(tmp_path: Path) -> None:
    _write_cache_file(tmp_path, "AAPL")
    out = download_ohlcv(
        ["AAPL"], start="2024-01-01", end="2024-01-31", cache_dir=tmp_path
    )
    assert not out.empty
    assert set(out.columns) == set(OHLCV_COLUMNS)
    assert (out["symbol"] == "AAPL").all()
    # All three rows fall inside the requested window.
    assert len(out) == 3
    # Date filter respects the upper bound exclusively.
    assert out["date"].min() >= pd.Timestamp("2024-01-01")
    assert out["date"].max() < pd.Timestamp("2024-01-31")


def test_download_respects_end_exclusive(tmp_path: Path) -> None:
    _write_cache_file(tmp_path, "MSFT")
    out = download_ohlcv(
        ["MSFT"], start="2024-01-02", end="2024-01-04", cache_dir=tmp_path
    )
    # Window [2024-01-02, 2024-01-04) excludes 2024-01-04 row.
    assert len(out) == 2
    assert out["date"].max() == pd.Timestamp("2024-01-03")


def test_download_drops_nan_rows(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["X", "X"],
            "open": [100.0, None],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1_000_000, 1_100_000],
            "adj_close": [100.5, 101.5],
        }
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp_path / "X.parquet", index=False)

    out = download_ohlcv(["X"], start="2024-01-01", end="2024-01-31", cache_dir=tmp_path)
    assert len(out) == 1
    assert out["date"].iloc[0] == pd.Timestamp("2024-01-02")
