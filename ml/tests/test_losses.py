"""Tests for the FinSight risk predictor losses."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from models import CombinedLoss, HuberLoss, PinballLoss, TruncatedMSE


def test_huber_basic() -> None:
    loss_fn: HuberLoss = HuberLoss(delta=0.01)

    pred: torch.Tensor = torch.tensor([0.05, 0.10, 0.20])
    target: torch.Tensor = torch.tensor([0.05, 0.10, 0.20])
    zero_loss: torch.Tensor = loss_fn(pred, target)
    assert torch.isclose(zero_loss, torch.tensor(0.0), atol=1e-7)

    # Off by exactly delta: huber reduces to 0.5 * delta**2 per element.
    delta: float = 0.01
    target_off: torch.Tensor = pred + delta
    near_quadratic: torch.Tensor = loss_fn(pred, target_off)
    expected: float = 0.5 * delta * delta
    assert torch.isclose(near_quadratic, torch.tensor(expected), atol=1e-9)


def test_pinball_asymmetric() -> None:
    """Underestimation (y_pred < y_true) must hurt more than overestimation at alpha=0.95.

    With magnitudes, y_true is a positive loss size. Underestimating it
    (predicting a smaller magnitude than realized) is the dangerous error.
    """
    loss_fn: PinballLoss = PinballLoss(alpha=0.95)

    y_true: torch.Tensor = torch.tensor([0.05])
    epsilon: float = 0.01

    # Under-prediction: pred below truth by epsilon.
    pred_under: torch.Tensor = torch.tensor([0.05 - epsilon])
    # Over-prediction: pred above truth by the same epsilon.
    pred_over: torch.Tensor = torch.tensor([0.05 + epsilon])

    loss_under: torch.Tensor = loss_fn(pred_under, y_true)
    loss_over: torch.Tensor = loss_fn(pred_over, y_true)

    assert loss_under > loss_over, (
        f"underestimation loss ({loss_under.item()}) should exceed "
        f"overestimation loss ({loss_over.item()}) at alpha=0.95"
    )

    # Ratio should be alpha / (1 - alpha) = 0.95 / 0.05 = 19.
    ratio: float = (loss_under / loss_over).item()
    assert abs(ratio - 19.0) < 1e-4, f"asymmetry ratio expected 19, got {ratio}"


def test_truncated_mse_zero_when_no_truncation() -> None:
    loss_fn: TruncatedMSE = TruncatedMSE()

    cvar_pred: torch.Tensor = torch.tensor([0.10, 0.20, 0.30], requires_grad=True)
    cvar_true: torch.Tensor = torch.tensor([0.05, 0.05, 0.05])
    var_true: torch.Tensor = torch.tensor([0.10, 0.10, 0.10])  # var >= cvar everywhere

    out: torch.Tensor = loss_fn(cvar_pred, cvar_true, var_true)
    assert torch.isclose(out, torch.tensor(0.0))

    # Zero scalar should still be a tensor with no NaN / Inf.
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_truncated_mse_filters() -> None:
    loss_fn: TruncatedMSE = TruncatedMSE()

    # Row 0: cvar_true (0.08) > var_true (0.05). Included. diff = pred - true = 0.10 - 0.08 = 0.02.
    # Row 1: cvar_true (0.04) <= var_true (0.05). Excluded.
    # Row 2: cvar_true (0.12) > var_true (0.10). Included. diff = 0.05 - 0.12 = -0.07.
    cvar_pred: torch.Tensor = torch.tensor([0.10, 0.06, 0.05])
    cvar_true: torch.Tensor = torch.tensor([0.08, 0.04, 0.12])
    var_true: torch.Tensor = torch.tensor([0.05, 0.05, 0.10])

    out: torch.Tensor = loss_fn(cvar_pred, cvar_true, var_true)

    expected: float = (0.02 ** 2 + (-0.07) ** 2) / 2.0
    assert torch.isclose(out, torch.tensor(expected), atol=1e-8), (
        f"truncated MSE expected {expected}, got {out.item()}"
    )


def test_combined_loss_components() -> None:
    torch.manual_seed(0)
    loss_fn: CombinedLoss = CombinedLoss(
        w_volatility=1.0,
        w_var=2.0,
        w_cvar=3.0,
    )

    pred: torch.Tensor = torch.tensor(
        [
            [0.15, 0.04, 0.06],
            [0.20, 0.05, 0.08],
        ]
    )
    target: torch.Tensor = torch.tensor(
        [
            [0.12, 0.05, 0.07],
            [0.22, 0.04, 0.09],
        ]
    )

    total, comps = loss_fn(pred, target)

    assert set(comps.keys()) == {"vol", "var", "cvar", "total"}

    expected_total: torch.Tensor = (
        1.0 * comps["vol"] + 2.0 * comps["var"] + 3.0 * comps["cvar"]
    )
    assert torch.isclose(total.detach(), expected_total, atol=1e-7), (
        f"total {total.item()} does not equal weighted sum {expected_total.item()}"
    )
    assert torch.isclose(comps["total"], expected_total, atol=1e-7)


def test_combined_loss_gradient() -> None:
    """Backprop through the combined loss must populate gradients on a stub net."""
    torch.manual_seed(0)

    # Stub net: 5-dim input, predicts 3 risk targets.
    net: nn.Sequential = nn.Sequential(
        nn.Linear(5, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )
    loss_fn: CombinedLoss = CombinedLoss()

    x: torch.Tensor = torch.randn(8, 5)
    target: torch.Tensor = torch.rand(8, 3).abs() * 0.1  # positive magnitudes

    pred: torch.Tensor = net(x)
    # Make var/cvar targets satisfy cvar > var so the truncated MSE contributes.
    target_fixed: torch.Tensor = target.clone()
    target_fixed[:, 2] = target_fixed[:, 1] + 0.01

    total, _ = loss_fn(pred, target_fixed)
    total.backward()

    nonzero_count: int = 0
    total_params: int = 0
    for p in net.parameters():
        assert p.grad is not None, "all stub-net parameters must have a gradient"
        assert torch.isfinite(p.grad).all(), "gradient must be finite"
        total_params += 1
        if torch.any(p.grad != 0):
            nonzero_count += 1
    assert nonzero_count == total_params, (
        f"expected gradients on all {total_params} param tensors, got {nonzero_count}"
    )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
