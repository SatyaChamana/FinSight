"""Portfolio-level feature engineering.

The model sees per-symbol features stacked side by side; these portfolio
features are extra channels that summarize portfolio-wide structure
(weighted returns, rolling volatility, concentration, average correlation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHT_TOLERANCE: float = 1e-6
CORRELATION_WINDOW: int = 63
PORTFOLIO_VOL_WINDOW: int = 21
ANNUALIZATION: float = float(np.sqrt(252.0))

PORTFOLIO_FEATURE_COLUMNS: list[str] = [
    "portfolio_return",
    "portfolio_vol_21",
    "portfolio_concentration",
    "portfolio_avg_correlation",
]


def _validate_weights(features: pd.DataFrame, weights: dict[str, float]) -> None:
    if not weights:
        raise ValueError("weights must be a non-empty dict")

    weight_sum = float(sum(weights.values()))
    if abs(weight_sum - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError(
            f"weights must sum to 1.0 within {WEIGHT_TOLERANCE}; got {weight_sum}"
        )

    present = set(features["symbol"].unique())
    missing = [s for s in weights if s not in present]
    if missing:
        raise ValueError(f"weights reference symbols not present in features: {missing}")


def _herfindahl(weights: dict[str, float]) -> float:
    return float(sum(w * w for w in weights.values()))


def _upper_tri_mean(corr: np.ndarray) -> float:
    """Mean of the strict upper triangle of a square correlation matrix."""
    n = corr.shape[0]
    if n < 2:
        return float("nan")
    iu = np.triu_indices(n, k=1)
    values = corr[iu]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def compute_portfolio_features(
    features: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    """Compute date-indexed portfolio features from per-symbol features.

    Args:
        features: Output of compute_features (long format with log_return).
        weights: Mapping symbol -> weight. Must sum to 1.0 within tolerance.

    Returns:
        DataFrame indexed by date with columns PORTFOLIO_FEATURE_COLUMNS.
    """
    _validate_weights(features, weights)

    symbols = list(weights.keys())
    # Pivot log_return into wide format: index = date, columns = symbol.
    returns_wide = (
        features[features["symbol"].isin(symbols)]
        .pivot(index="date", columns="symbol", values="log_return")
        .sort_index()
    )

    # Restrict to dates where every symbol has a return (alignment).
    returns_wide = returns_wide.dropna(how="any")
    returns_wide = returns_wide[symbols]

    weight_vec = np.array([weights[s] for s in symbols], dtype=np.float64)
    portfolio_return = pd.Series(
        returns_wide.values @ weight_vec,
        index=returns_wide.index,
        name="portfolio_return",
    )

    portfolio_vol_21 = (
        portfolio_return.rolling(window=PORTFOLIO_VOL_WINDOW, min_periods=PORTFOLIO_VOL_WINDOW)
        .std()
        * ANNUALIZATION
    )

    concentration = _herfindahl(weights)

    # Rolling pairwise correlation: take mean of upper triangle at each date.
    avg_corr_values: list[float] = []
    dates = returns_wide.index
    arr = returns_wide.values  # shape [T, P]
    n_dates = arr.shape[0]
    for i in range(n_dates):
        if i + 1 < CORRELATION_WINDOW:
            avg_corr_values.append(float("nan"))
            continue
        window = arr[i + 1 - CORRELATION_WINDOW : i + 1]
        # Need variation to compute correlation; if any column is constant we
        # fall back to NaN for that pair, then take the mean of the rest.
        if window.shape[0] < 2:
            avg_corr_values.append(float("nan"))
            continue
        corr = np.corrcoef(window, rowvar=False)
        avg_corr_values.append(_upper_tri_mean(corr))

    avg_corr_series = pd.Series(avg_corr_values, index=dates, name="portfolio_avg_correlation")

    out = pd.DataFrame(
        {
            "portfolio_return": portfolio_return,
            "portfolio_vol_21": portfolio_vol_21,
            "portfolio_concentration": concentration,
            "portfolio_avg_correlation": avg_corr_series,
        },
        index=dates,
    )
    out.index.name = "date"
    return out[PORTFOLIO_FEATURE_COLUMNS]
