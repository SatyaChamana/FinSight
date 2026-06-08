"""PyTorch Dataset for portfolio risk training.

Each item is a (lookback_days, F)-shaped feature window paired with a 3-vector
of forward-looking risk targets (volatility, VaR95, CVaR95) computed over the
next horizon_days of portfolio returns.

Upstream feature engineering (compute_features, compute_portfolio_features)
is responsible for dropping NaN warm-up rows. This class assumes its inputs
are already finite on the dates it consumes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.features import FEATURE_COLUMNS
from data.portfolio_features import PORTFOLIO_FEATURE_COLUMNS

ANNUALIZATION: float = float(np.sqrt(252.0))


class PortfolioRiskDataset(Dataset):
    """Windowed dataset over aligned per-symbol + portfolio features.

    Layout of x at index i:
        rows: lookback_days consecutive trading days ending at date d_i
        cols: [s0_f0, s0_f1, ..., sP_fF, p0, p1, ...] where s* are per-symbol
              feature blocks (one block per symbol, in `symbols` order) and p*
              are the portfolio features in PORTFOLIO_FEATURE_COLUMNS order.

    The dataset length is the number of windows for which a full forward
    horizon of portfolio returns is also available, i.e. len(dates)
    - lookback_days - horizon_days + 1.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        portfolio_features: pd.DataFrame,
        weights: dict[str, float],
        lookback_days: int = 63,
        horizon_days: int = 21,
        feature_cols: list[str] | None = None,
    ) -> None:
        if lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")
        if horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")

        self.lookback_days = lookback_days
        self.horizon_days = horizon_days
        self.symbols: list[str] = list(weights.keys())
        self.feature_cols: list[str] = list(feature_cols) if feature_cols is not None else list(FEATURE_COLUMNS)

        # Build a per-symbol date-indexed frame restricted to feature_cols.
        # Then align all symbols on the same date axis (intersection of dates).
        per_symbol_frames: dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            sub = features.loc[features["symbol"] == symbol, ["date"] + self.feature_cols]
            sub = sub.set_index("date").sort_index()
            per_symbol_frames[symbol] = sub

        # Intersection of date indices across symbols and portfolio features.
        common_dates = portfolio_features.index
        for sub in per_symbol_frames.values():
            common_dates = common_dates.intersection(sub.index)
        common_dates = common_dates.sort_values()

        if len(common_dates) < lookback_days + horizon_days:
            raise ValueError(
                "not enough aligned dates: need at least "
                f"{lookback_days + horizon_days}, got {len(common_dates)}"
            )

        # Stack per-symbol features side by side: shape [T, P * F].
        symbol_blocks: list[np.ndarray] = []
        for symbol in self.symbols:
            block = per_symbol_frames[symbol].loc[common_dates, self.feature_cols].to_numpy(dtype=np.float32)
            symbol_blocks.append(block)
        per_symbol_matrix = np.concatenate(symbol_blocks, axis=1)  # [T, P*F]

        portfolio_matrix = (
            portfolio_features.loc[common_dates, PORTFOLIO_FEATURE_COLUMNS]
            .to_numpy(dtype=np.float32)
        )  # [T, n_portfolio_features]

        self._feature_matrix: np.ndarray = np.concatenate(
            [per_symbol_matrix, portfolio_matrix], axis=1
        )  # [T, F_total]

        self._portfolio_returns: np.ndarray = (
            portfolio_features.loc[common_dates, "portfolio_return"].to_numpy(dtype=np.float32)
        )
        self._dates: pd.DatetimeIndex = pd.DatetimeIndex(common_dates)

        self._n_symbols: int = len(self.symbols)
        self._n_per_symbol_features: int = len(self.feature_cols)
        self._n_portfolio_features: int = len(PORTFOLIO_FEATURE_COLUMNS)

    @property
    def feature_dim(self) -> int:
        return self._feature_matrix.shape[1]

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self._dates

    def __len__(self) -> int:
        return max(0, len(self._dates) - self.lookback_days - self.horizon_days + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        lb_start = idx
        lb_end = idx + self.lookback_days  # exclusive
        horizon_end = lb_end + self.horizon_days  # exclusive

        window = self._feature_matrix[lb_start:lb_end]
        forward_returns = self._portfolio_returns[lb_end:horizon_end]

        volatility = float(np.std(forward_returns, ddof=0) * ANNUALIZATION)

        # 5th percentile of returns is a (typically negative) loss; convert to
        # positive magnitude. linear interpolation is the numpy default.
        q05 = float(np.quantile(forward_returns, 0.05))
        var_95 = float(-q05)
        if var_95 < 0.0:
            var_95 = 0.0  # tail can land on the positive side for short horizons

        tail_mask = forward_returns <= q05
        tail_values = forward_returns[tail_mask]
        if tail_values.size == 0:
            cvar_95 = var_95
        else:
            cvar_95 = float(-tail_values.mean())
            if cvar_95 < 0.0:
                cvar_95 = 0.0

        x = torch.from_numpy(window.astype(np.float32, copy=False))
        y = torch.tensor([volatility, var_95, cvar_95], dtype=torch.float32)
        return x, y
