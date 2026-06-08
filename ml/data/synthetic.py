"""Synthetic stress-scenario OHLCV generation.

Each scenario takes a baseline OHLCV frame (long format from download_ohlcv)
and applies a named perturbation so the same downstream feature pipeline can
run on it. severity scales the perturbation magnitude (severity=0 returns
the base unchanged; severity=1 is the "named" intensity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCENARIOS: tuple[str, ...] = (
    "flash_crash",
    "vol_regime_shift",
    "circuit_breaker",
    "correlation_spike",
)


def _validate_base(base_ohlcv: pd.DataFrame) -> None:
    required = {"date", "symbol", "open", "high", "low", "close", "volume", "adj_close"}
    missing = required - set(base_ohlcv.columns)
    if missing:
        raise ValueError(f"base_ohlcv missing required columns: {sorted(missing)}")


def _apply_price_multiplier(
    df: pd.DataFrame, mask: pd.Series, multiplier: pd.Series | float
) -> pd.DataFrame:
    """Multiply open/high/low/close/adj_close by multiplier on the masked rows."""
    out = df.copy()
    for col in ("open", "high", "low", "close", "adj_close"):
        if isinstance(multiplier, pd.Series):
            mult = multiplier.reindex(out.index).fillna(1.0)
            out.loc[mask, col] = out.loc[mask, col] * mult.loc[mask]
        else:
            out.loc[mask, col] = out.loc[mask, col] * multiplier
    return out


def _flash_crash(df: pd.DataFrame, severity: float) -> pd.DataFrame:
    """Single sharp drop on the midpoint date (per symbol)."""
    drop_pct = 0.10 * severity  # severity=1 => 10% intraday drop
    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby("symbol", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        if group.empty:
            pieces.append(group)
            continue
        crash_idx = len(group) // 2
        mult = 1.0 - drop_pct
        group.loc[crash_idx, ["close", "adj_close", "low"]] = (
            group.loc[crash_idx, ["close", "adj_close", "low"]].astype(float) * mult
        )
        # Propagate the lower price level forward so subsequent rows reflect
        # the crash rather than reverting to baseline.
        if crash_idx + 1 < len(group):
            group.loc[crash_idx + 1 :, ["open", "high", "low", "close", "adj_close"]] = (
                group.loc[crash_idx + 1 :, ["open", "high", "low", "close", "adj_close"]].astype(float)
                * mult
            )
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def _vol_regime_shift(df: pd.DataFrame, severity: float) -> pd.DataFrame:
    """Double (or severity-scaled) daily-return volatility over second half."""
    vol_mult = 1.0 + 1.0 * severity  # severity=1 => 2x daily moves
    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby("symbol", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        n = len(group)
        if n < 2:
            pieces.append(group)
            continue
        midpoint = n // 2

        # Compute daily log returns from adj_close, then amplify second-half
        # returns and rebuild the price series so OHLC and adj_close stay
        # internally consistent.
        adj = group["adj_close"].astype(float).to_numpy()
        log_ret = np.zeros(n, dtype=np.float64)
        log_ret[1:] = np.log(adj[1:] / adj[:-1])

        mean_second_half = float(log_ret[midpoint:].mean()) if n - midpoint > 0 else 0.0
        amplified = log_ret.copy()
        amplified[midpoint:] = mean_second_half + (log_ret[midpoint:] - mean_second_half) * vol_mult

        new_adj = np.empty(n, dtype=np.float64)
        new_adj[0] = adj[0]
        new_adj[1:] = adj[0] * np.exp(np.cumsum(amplified[1:]))

        scale = new_adj / np.where(adj == 0.0, 1.0, adj)
        group["adj_close"] = new_adj
        for col in ("open", "high", "low", "close"):
            group[col] = group[col].astype(float).to_numpy() * scale

        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def _circuit_breaker(df: pd.DataFrame, severity: float) -> pd.DataFrame:
    """Force a contiguous stretch of flat days (no price change, zero volume)."""
    flat_days = max(1, int(round(5 * severity)))  # severity=1 => 5 flat days
    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby("symbol", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        n = len(group)
        if n == 0:
            pieces.append(group)
            continue
        start = max(0, n // 2 - flat_days // 2)
        end = min(n, start + flat_days)
        if end > start:
            ref_close = float(group.loc[start, "close"])
            ref_adj = float(group.loc[start, "adj_close"])
            for col in ("open", "high", "low", "close"):
                group.loc[start:end - 1, col] = ref_close
            group.loc[start:end - 1, "adj_close"] = ref_adj
            group.loc[start:end - 1, "volume"] = 0
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def _correlation_spike(df: pd.DataFrame, severity: float) -> pd.DataFrame:
    """Collapse all symbols toward a single shared return factor.

    severity=0 keeps original returns, severity=1 forces all symbols onto the
    average return path (correlation -> 1.0).
    """
    if df.empty:
        return df

    symbols = list(df["symbol"].unique())
    pieces: list[pd.DataFrame] = []

    # Compute per-symbol returns aligned on a wide frame so we can average.
    wide_pieces: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        sub = df[df["symbol"] == symbol].sort_values("date").reset_index(drop=True)
        adj = sub["adj_close"].astype(float).to_numpy()
        n = len(adj)
        log_ret = np.zeros(n, dtype=np.float64)
        if n > 1:
            log_ret[1:] = np.log(adj[1:] / adj[:-1])
        wide_pieces[symbol] = pd.DataFrame(
            {"date": sub["date"].values, "log_return": log_ret, "adj_close_0": [adj[0]] * n}
        )

    # Build the common factor: mean log-return across symbols on each date.
    merged = None
    for symbol, frame in wide_pieces.items():
        col = frame.rename(columns={"log_return": f"ret_{symbol}"})[["date", f"ret_{symbol}"]]
        merged = col if merged is None else merged.merge(col, on="date", how="outer")
    assert merged is not None
    merged = merged.sort_values("date").reset_index(drop=True)
    ret_cols = [c for c in merged.columns if c.startswith("ret_")]
    factor = merged[ret_cols].mean(axis=1).fillna(0.0).to_numpy()
    factor_by_date = dict(zip(merged["date"].to_numpy(), factor, strict=True))

    for symbol in symbols:
        sub = df[df["symbol"] == symbol].sort_values("date").reset_index(drop=True).copy()
        adj0 = float(sub["adj_close"].iloc[0])
        adj = sub["adj_close"].astype(float).to_numpy()
        n = len(adj)
        log_ret = np.zeros(n, dtype=np.float64)
        if n > 1:
            log_ret[1:] = np.log(adj[1:] / adj[:-1])

        factor_series = np.array(
            [factor_by_date.get(d, 0.0) for d in sub["date"].to_numpy()], dtype=np.float64
        )
        new_ret = (1.0 - severity) * log_ret + severity * factor_series

        new_adj = np.empty(n, dtype=np.float64)
        new_adj[0] = adj0
        if n > 1:
            new_adj[1:] = adj0 * np.exp(np.cumsum(new_ret[1:]))

        scale = new_adj / np.where(adj == 0.0, 1.0, adj)
        sub["adj_close"] = new_adj
        for col in ("open", "high", "low", "close"):
            sub[col] = sub[col].astype(float).to_numpy() * scale

        pieces.append(sub)

    return pd.concat(pieces, ignore_index=True)


def generate_stress_scenario(
    base_ohlcv: pd.DataFrame,
    scenario: str,
    severity: float = 1.0,
) -> pd.DataFrame:
    """Return a perturbed OHLCV frame matching the schema of download_ohlcv.

    Args:
        base_ohlcv: Long-format OHLCV (output of download_ohlcv).
        scenario: One of SCENARIOS.
        severity: Scales the perturbation. 0.0 returns base unchanged.
    """
    _validate_base(base_ohlcv)
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")

    if severity == 0.0:
        return base_ohlcv.copy().reset_index(drop=True)

    if scenario == "flash_crash":
        out = _flash_crash(base_ohlcv, severity)
    elif scenario == "vol_regime_shift":
        out = _vol_regime_shift(base_ohlcv, severity)
    elif scenario == "circuit_breaker":
        out = _circuit_breaker(base_ohlcv, severity)
    elif scenario == "correlation_spike":
        out = _correlation_spike(base_ohlcv, severity)
    else:  # pragma: no cover, guarded above
        raise ValueError(scenario)

    return out.sort_values(["symbol", "date"]).reset_index(drop=True)
