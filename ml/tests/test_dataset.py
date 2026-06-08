"""Tests for data.dataset.PortfolioRiskDataset."""

from __future__ import annotations

import pandas as pd
import torch

from data.dataset import PortfolioRiskDataset
from data.features import FEATURE_COLUMNS, compute_features
from data.portfolio_features import (
    PORTFOLIO_FEATURE_COLUMNS,
    compute_portfolio_features,
)


def _build_pipeline(ohlcv: pd.DataFrame, weights: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = compute_features(ohlcv)
    port = compute_portfolio_features(feats, weights)
    # Drop NaNs from rolling vol / correlation warm-up at portfolio level so
    # the dataset receives finite values everywhere.
    port = port.dropna()
    feats = feats[feats["date"].isin(port.index)].reset_index(drop=True)
    return feats, port


def test_dataset_shapes(synthetic_ohlcv_three_symbols: pd.DataFrame) -> None:
    weights = {"AAA": 0.4, "BBB": 0.3, "CCC": 0.3}
    feats, port = _build_pipeline(synthetic_ohlcv_three_symbols, weights)
    ds = PortfolioRiskDataset(
        features=feats,
        portfolio_features=port,
        weights=weights,
        lookback_days=63,
        horizon_days=21,
    )
    assert len(ds) == len(port.index) - 63 - 21 + 1

    x, y = ds[0]
    expected_feature_dim = len(weights) * len(FEATURE_COLUMNS) + len(PORTFOLIO_FEATURE_COLUMNS)
    assert x.shape == (63, expected_feature_dim)
    assert y.shape == (3,)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_dataset_targets_are_finite(synthetic_ohlcv_three_symbols: pd.DataFrame) -> None:
    weights = {"AAA": 0.5, "BBB": 0.25, "CCC": 0.25}
    feats, port = _build_pipeline(synthetic_ohlcv_three_symbols, weights)
    ds = PortfolioRiskDataset(
        features=feats,
        portfolio_features=port,
        weights=weights,
        lookback_days=30,
        horizon_days=10,
    )
    for i in range(len(ds)):
        x, y = ds[i]
        assert torch.isfinite(x).all()
        assert torch.isfinite(y).all()
        # Volatility and VaR/CVaR magnitudes are non-negative.
        assert (y >= 0.0).all()


def test_dataset_respects_custom_feature_cols(
    synthetic_ohlcv_three_symbols: pd.DataFrame,
) -> None:
    weights = {"AAA": 0.4, "BBB": 0.3, "CCC": 0.3}
    feats, port = _build_pipeline(synthetic_ohlcv_three_symbols, weights)
    custom = ["log_return", "rolling_vol_21"]
    ds = PortfolioRiskDataset(
        features=feats,
        portfolio_features=port,
        weights=weights,
        lookback_days=30,
        horizon_days=10,
        feature_cols=custom,
    )
    x, _ = ds[0]
    expected = len(weights) * len(custom) + len(PORTFOLIO_FEATURE_COLUMNS)
    assert x.shape[1] == expected
