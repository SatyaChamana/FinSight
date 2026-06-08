"""Tests for the RiskPredictor architecture."""

from __future__ import annotations

import pathlib

import onnx
import pytest
import torch

from models import RiskPredictor


def _make_model(input_dim: int = 50) -> RiskPredictor:
    torch.manual_seed(0)
    return RiskPredictor(input_dim=input_dim)


def test_forward_shape() -> None:
    model: RiskPredictor = _make_model(input_dim=50)
    x: torch.Tensor = torch.randn(4, 63, 50)

    out: torch.Tensor = model(x)

    assert out.shape == (4, 3)
    assert torch.isfinite(out).all(), "forward produced NaN or Inf"

    # Gradient must flow.
    out.sum().backward()
    grads_present: bool = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )
    assert grads_present, "no gradients flowed during backward"


def test_batch_independence() -> None:
    model: RiskPredictor = _make_model(input_dim=50)
    model.eval()

    torch.manual_seed(42)
    x_batch: torch.Tensor = torch.randn(2, 63, 50)

    with torch.no_grad():
        out_batch: torch.Tensor = model(x_batch)
        out_single: torch.Tensor = model(x_batch[:1])

    assert torch.allclose(out_batch[0], out_single[0], atol=1e-5), (
        "first element of batch=2 forward must match the batch=1 forward (no cross-batch leakage)"
    )


def test_parameter_count() -> None:
    model: RiskPredictor = _make_model(input_dim=50)
    n_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Default config (input=50, hidden=128, 2 LSTM layers, 4 attention heads)
    # produces roughly 250k parameters. The looser lower bound here is
    # intentional so unrelated config tweaks don't break the test.
    assert n_params > 100_000, f"parameter count {n_params} is unexpectedly small"


def test_onnx_exportable(tmp_path: pathlib.Path) -> None:
    model: RiskPredictor = _make_model(input_dim=50)
    model.eval()

    dummy: torch.Tensor = torch.randn(1, 63, 50)
    out_path: pathlib.Path = tmp_path / "risk_predictor.onnx"

    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        opset_version=17,
        dynamo=False,
        input_names=["features"],
        output_names=["risk"],
        dynamic_axes={"features": {0: "batch"}, "risk": {0: "batch"}},
    )

    assert out_path.exists(), "ONNX export did not write a file"

    loaded: onnx.ModelProto = onnx.load(str(out_path))
    onnx.checker.check_model(loaded)


def test_output_columns() -> None:
    model: RiskPredictor = _make_model(input_dim=50)
    assert model.output_columns == ("volatility", "var_95", "cvar_95")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
