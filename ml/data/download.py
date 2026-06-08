"""OHLCV download with on-disk parquet caching.

Returns long-format DataFrames (one row per (date, symbol)) so the downstream
feature pipeline can groupby('symbol') without reshaping.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical schema returned by download_ohlcv. Anything reading the output of
# this function (features, synthetic, dataset) assumes exactly these columns.
OHLCV_COLUMNS: list[str] = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
]


def _normalize_symbol_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize a yfinance result into long-format OHLCV for one symbol."""
    if df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    # yfinance can return a MultiIndex column when downloading for a list of
    # tickers. We always pass a single ticker here, but the user might pass
    # group_by; flatten defensively.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index().rename(
        columns={
            "Date": "date",
            "Datetime": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
        }
    )

    # Some yfinance versions inline auto-adjusted close. Fall back to close if
    # no separate adj_close column exists.
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

    df = df[OHLCV_COLUMNS]
    df = df.dropna(subset=["open", "high", "low", "close", "volume", "adj_close"])
    return df.reset_index(drop=True)


def _load_cached(symbol: str, cache_dir: Path) -> pd.DataFrame | None:
    path = cache_dir / f"{symbol}.parquet"
    if not path.exists():
        return None
    logger.info("cache hit for %s at %s", symbol, path)
    df = pd.read_parquet(path)
    # Ensure cached frames still conform to schema.
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("cached frame for %s missing columns %s; refetching", symbol, missing)
        return None
    return df[OHLCV_COLUMNS]


def _write_cache(df: pd.DataFrame, symbol: str, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{symbol}.parquet"
    df.to_parquet(path, index=False)


def download_ohlcv(
    symbols: list[str],
    start: str,
    end: str,
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV for each symbol from yfinance.

    Args:
        symbols: List of ticker symbols (e.g. ["AAPL", "MSFT"]).
        start: ISO date string, inclusive lower bound.
        end: ISO date string, exclusive upper bound (yfinance convention).
        cache_dir: Optional directory for per-symbol parquet caches. If a cache
            file exists, it is loaded instead of hitting yfinance.

    Returns:
        Long-format DataFrame with columns OHLCV_COLUMNS, sorted by
        (symbol, date). Date is timezone-naive (date-only, midnight UTC).
    """
    if not symbols:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    cache_path = Path(cache_dir) if cache_dir is not None else None

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        df: pd.DataFrame | None = None
        if cache_path is not None:
            df = _load_cached(symbol, cache_path)

        if df is None:
            # Import yfinance lazily so tests that only hit the cache path do
            # not require a network. yfinance also touches the network at
            # import time on some versions.
            import yfinance as yf

            raw = yf.download(
                symbol,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                group_by="column",
            )
            df = _normalize_symbol_frame(raw, symbol)
            if cache_path is not None and not df.empty:
                _write_cache(df, symbol, cache_path)

        if df is not None and not df.empty:
            # Filter to requested date range (cache may have wider coverage).
            start_ts = pd.to_datetime(start)
            end_ts = pd.to_datetime(end)
            mask = (df["date"] >= start_ts) & (df["date"] < end_ts)
            df = df.loc[mask]
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["open", "high", "low", "close", "volume", "adj_close"])
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    return out
