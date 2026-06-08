"""FinSight ML data pipeline.

Public API for the Phase 4 data layer: OHLCV download, per-symbol feature
engineering, portfolio-level features, walk-forward splitting, the PyTorch
dataset, and synthetic stress scenarios.
"""

from data.dataset import PortfolioRiskDataset
from data.download import download_ohlcv
from data.features import compute_features
from data.portfolio_features import compute_portfolio_features
from data.scaler import FeatureScaler
from data.split import walk_forward_split
from data.synthetic import generate_stress_scenario

__all__ = [
    "FeatureScaler",
    "PortfolioRiskDataset",
    "compute_features",
    "compute_portfolio_features",
    "download_ohlcv",
    "generate_stress_scenario",
    "walk_forward_split",
]
