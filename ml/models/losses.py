"""Loss components for the FinSight risk predictor.

Sign convention for VaR-95 and CVaR-95 (used everywhere in this module and
downstream):

    Both targets and predictions are expressed as POSITIVE MAGNITUDES of the
    tail loss. That is, VaR-95 = 0.04 means the 5th percentile loss is 4
    percent of portfolio value. CVaR-95 >= VaR-95 by construction (the
    expected loss conditional on being in the worst 5 percent is at least as
    bad as the threshold). Volatility is also a positive magnitude (the
    realized standard deviation of returns).

    This convention is important for the pinball loss: with magnitudes,
    UNDERESTIMATING the loss (y_pred < y_true) is the dangerous error, so
    that side of the asymmetry gets the larger weight alpha. The signed
    quantile-regression formula has been flipped to honor that.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class HuberLoss(nn.Module):
    """Huber loss with mean reduction. Used for the volatility head.

    Quadratic when |y_pred - y| <= delta, linear beyond delta. With the
    default delta of 0.01 (1 percent volatility), small errors are penalized
    smoothly and larger outliers do not dominate the gradient.
    """

    def __init__(self, delta: float = 0.01) -> None:
        super().__init__()
        self.delta: float = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(pred, target, reduction="mean", delta=self.delta)


class PinballLoss(nn.Module):
    """Asymmetric quantile (pinball) loss for the VaR-95 head.

    alpha is the VaR confidence level (default 0.95). With the positive
    magnitude convention documented at the top of this module, the loss is:

        L = mean(
            alpha       * max(y - y_pred, 0)   # underestimate penalty
            + (1 - alpha) * max(y_pred - y, 0) # overestimate penalty
        )

    Underestimating tail risk (y_pred < y) carries the heavier alpha weight,
    which is what we want for a risk metric.

    Note on the spec's "q = 1 - alpha = 0.05" comment: the standard
    signed-return formula uses q = 0.05 as the quantile parameter. When the
    variables are flipped to magnitudes, the roles of q and (1 - q) swap, so
    the heavier weight ends up being alpha rather than q. The asymmetry
    direction is preserved.
    """

    def __init__(self, alpha: float = 0.95) -> None:
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha: float = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error: torch.Tensor = target - pred
        # Underestimate side (error > 0) carries weight alpha.
        # Overestimate side (error < 0) carries weight (1 - alpha).
        under: torch.Tensor = torch.clamp(error, min=0.0)
        over: torch.Tensor = torch.clamp(-error, min=0.0)
        loss: torch.Tensor = self.alpha * under + (1.0 - self.alpha) * over
        return loss.mean()


class TruncatedMSE(nn.Module):
    """MSE on CVaR predictions, restricted to rows where cvar_true > var_true.

    The truncation enforces the structural constraint CVaR > VaR: only rows
    that actually satisfy the inequality contribute to the loss. When no
    rows satisfy it (rare, mostly during early training on noisy batches),
    a zero scalar is returned with the same device and dtype, preserving
    gradient flow through the rest of the network.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        cvar_pred: torch.Tensor,
        cvar_true: torch.Tensor,
        var_true: torch.Tensor,
    ) -> torch.Tensor:
        mask: torch.Tensor = cvar_true > var_true
        if not torch.any(mask):
            return torch.zeros((), device=cvar_pred.device, dtype=cvar_pred.dtype)

        diff: torch.Tensor = (cvar_pred - cvar_true)[mask]
        return (diff * diff).mean()


class CombinedLoss(nn.Module):
    """Weighted combination of the three head-specific losses.

    Forward expects pred and target of shape [batch, 3] where columns are
    (volatility, var_95, cvar_95) in that order. Returns the scalar total
    plus a dictionary of detached component scalars suitable for logging.
    """

    def __init__(
        self,
        w_volatility: float = 1.0,
        w_var: float = 1.0,
        w_cvar: float = 1.0,
        huber_delta: float = 0.01,
        pinball_alpha: float = 0.95,
    ) -> None:
        super().__init__()
        self.w_volatility: float = w_volatility
        self.w_var: float = w_var
        self.w_cvar: float = w_cvar

        self.huber: HuberLoss = HuberLoss(delta=huber_delta)
        self.pinball: PinballLoss = PinballLoss(alpha=pinball_alpha)
        self.truncated_mse: TruncatedMSE = TruncatedMSE()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        vol_pred: torch.Tensor = pred[:, 0]
        var_pred: torch.Tensor = pred[:, 1]
        cvar_pred: torch.Tensor = pred[:, 2]

        vol_true: torch.Tensor = target[:, 0]
        var_true: torch.Tensor = target[:, 1]
        cvar_true: torch.Tensor = target[:, 2]

        loss_vol: torch.Tensor = self.huber(vol_pred, vol_true)
        loss_var: torch.Tensor = self.pinball(var_pred, var_true)
        loss_cvar: torch.Tensor = self.truncated_mse(cvar_pred, cvar_true, var_true)

        total: torch.Tensor = (
            self.w_volatility * loss_vol
            + self.w_var * loss_var
            + self.w_cvar * loss_cvar
        )

        components: dict[str, torch.Tensor] = {
            "vol": loss_vol.detach(),
            "var": loss_var.detach(),
            "cvar": loss_cvar.detach(),
            "total": total.detach(),
        }
        return total, components
