"""Per-symbol feature engineering.

All features are computed per-symbol then concatenated. Rolling volatilities
are annualized (multiplied by sqrt(252)) so they live on the same scale as
the model targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALIZATION: float = float(np.sqrt(252.0))

FEATURE_COLUMNS: list[str] = [
    "log_return",
    "rolling_vol_5",
    "rolling_vol_21",
    "rolling_vol_63",
    "ewma_vol_21",
    "garman_klass_vol",
    "rsi_14",
    "atr_14",
    "bb_width_20",
]


def _log_return(adj_close: pd.Series) -> pd.Series:
    return np.log(adj_close / adj_close.shift(1))


def _rolling_vol(log_ret: pd.Series, window: int) -> pd.Series:
    return log_ret.rolling(window=window, min_periods=window).std() * ANNUALIZATION


def _ewma_vol(log_ret: pd.Series, halflife: int) -> pd.Series:
    return log_ret.ewm(halflife=halflife, adjust=False).std() * ANNUALIZATION


def _garman_klass(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Garman-Klass volatility estimator (annualized).

    Per-day variance estimate:
        0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2
    """
    hl = np.log(high / low)
    co = np.log(close / open_)
    daily_var = 0.5 * hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * co.pow(2)
    # Variance can go slightly negative on flat days due to floating noise;
    # clip to zero before sqrt.
    daily_var = daily_var.clip(lower=0.0)
    return np.sqrt(daily_var) * ANNUALIZATION


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI using exponential smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing is an EMA with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss is zero the asset has only gone up; RSI saturates at 100.
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _bb_width(close: pd.Series, window: int = 20) -> pd.Series:
    """Bollinger Band width, expressed in standard-deviation units.

    width = (upper - lower) / sma = 4 * std / sma (since upper/lower = sma +/- 2*std).
    """
    sma = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    return (4.0 * std) / sma.replace(0.0, np.nan)


def _features_for_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for a single symbol's OHLCV frame, sorted by date."""
    df = df.sort_values("date").reset_index(drop=True).copy()

    df["log_return"] = _log_return(df["adj_close"])
    df["rolling_vol_5"] = _rolling_vol(df["log_return"], 5)
    df["rolling_vol_21"] = _rolling_vol(df["log_return"], 21)
    df["rolling_vol_63"] = _rolling_vol(df["log_return"], 63)
    df["ewma_vol_21"] = _ewma_vol(df["log_return"], halflife=21)
    df["garman_klass_vol"] = _garman_klass(df["open"], df["high"], df["low"], df["close"])
    df["rsi_14"] = _rsi(df["close"], period=14)
    df["atr_14"] = _atr(df["high"], df["low"], df["close"], period=14)
    df["bb_width_20"] = _bb_width(df["close"], window=20)

    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return df


def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute per-symbol features over long-format OHLCV.

    Args:
        ohlcv: Output of download_ohlcv (long format).

    Returns:
        Same long-format frame plus FEATURE_COLUMNS, with warm-up rows dropped
        per symbol. Sorted by (symbol, date).
    """
    if ohlcv.empty:
        return ohlcv.assign(**{col: pd.Series(dtype="float64") for col in FEATURE_COLUMNS})

    pieces: list[pd.DataFrame] = []
    for _, group in ohlcv.groupby("symbol", sort=False):
        pieces.append(_features_for_symbol(group))

    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)
