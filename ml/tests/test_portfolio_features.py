"""Tests for data.portfolio_features."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from data.features import compute_features
from data.portfolio_features import (
    PORTFOLIO_FEATURE_COLUMNS,
    compute_portfolio_features,
)


def test_portfolio_features_basic_shape(synthetic_ohlcv_three_symbols: pd.DataFrame) -> None:
    feats = compute_features(synthetic_ohlcv_three_symbols)
    weights = {"AAA": 0.4, "BBB": 0.3, "CCC": 0.3}
    out = compute_portfolio_features(feats, weights)
    assert list(out.columns) == PORTFOLIO_FEATURE_COLUMNS
    assert out.index.name == "date"
    assert not out.empty


def test_portfolio_concentration_is_constant(synthetic_ohlcv_three_symbols: pd.DataFrame) -> None:
    feats = compute_features(synthetic_ohlcv_three_symbols)
    weights = {"AAA": 0.4, "BBB": 0.3, "CCC": 0.3}
    out = compute_portfolio_features(feats, weights)
    expected = 0.4 ** 2 + 0.3 ** 2 + 0.3 ** 2
    assert (out["portfolio_concentration"] == out["portfolio_concentration"].iloc[0]).all()
    assert math.isclose(float(out["portfolio_concentration"].iloc[0]), expected, rel_tol=1e-12)


def test_portfolio_return_is_weighted_combination(synthetic_ohlcv_three_symbols: pd.DataFrame) -> None:
    feats = compute_features(synthetic_ohlcv_three_symbols)
    weights = {"AAA": 0.5, "BBB": 0.25, "CCC": 0.25}
    port = compute_portfolio_features(feats, weights)

    wide = (
        feats[feats["symbol"].isin(weights)]
        .pivot(index="date", columns="symbol", values="log_return")
        .dropna(how="any")
    )
    expected = (
        0.5 * wide["AAA"] + 0.25 * wide["BBB"] + 0.25 * wide["CCC"]
    ).loc[port.index]
    assert (port["portfolio_return"] - expected).abs().max() < 1e-12


def test_portfolio_features_rejects_non_unit_weights(
    synthetic_ohlcv_three_symbols: pd.DataFrame,
) -> None:
    feats = compute_features(synthetic_ohlcv_three_symbols)
    with pytest.raises(ValueError):
        compute_portfolio_features(feats, {"AAA": 0.5, "BBB": 0.4})


def test_portfolio_features_rejects_unknown_symbols(
    synthetic_ohlcv_three_symbols: pd.DataFrame,
) -> None:
    feats = compute_features(synthetic_ohlcv_three_symbols)
    with pytest.raises(ValueError):
        compute_portfolio_features(feats, {"AAA": 0.5, "ZZZ": 0.5})


def test_portfolio_avg_correlation_in_unit_interval(
    synthetic_ohlcv_three_symbols: pd.DataFrame,
) -> None:
    feats = compute_features(synthetic_ohlcv_three_symbols)
    weights = {"AAA": 0.4, "BBB": 0.3, "CCC": 0.3}
    out = compute_portfolio_features(feats, weights).dropna(subset=["portfolio_avg_correlation"])
    assert ((out["portfolio_avg_correlation"] >= -1.0) & (out["portfolio_avg_correlation"] <= 1.0)).all()
