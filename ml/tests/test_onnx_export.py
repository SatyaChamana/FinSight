"""Tests for ``export.onnx_export``.

The tests use a deliberately tiny PyTorch model so they run standalone, without
depending on the real ``RiskPredictor`` architecture or any feature pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn

from export.onnx_export import export_model, validate_onnx_parity

INPUT_DIM = 8
SEQ_LEN = 5


class TinyModel(nn.Module):
    """Toy [B, T, F] -> [B, 3] model for export tests."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mean-pool over the time dimension, then two dense layers.
        return self.fc2(torch.relu(self.fc1(x.mean(dim=1))))


class DivergentModel(nn.Module):
    """Same I/O contract as ``TinyModel`` but a structurally different graph.

    Used to drive the parity check off a cliff so we can assert that
    ``validate_onnx_parity`` raises.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Different reduction (sum vs mean) and a different non-linearity (tanh
        # vs relu) guarantee numerical divergence from the exported TinyModel.
        return self.fc2(torch.tanh(self.fc1(x.sum(dim=1) * 7.5)))


@pytest.fixture
def toy_model() -> TinyModel:
    torch.manual_seed(0)
    return TinyModel(INPUT_DIM)


@pytest.fixture
def sample_input() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, SEQ_LEN, INPUT_DIM)


def test_export_writes_file(toy_model: TinyModel, sample_input: torch.Tensor, tmp_path: Path) -> None:
    out = tmp_path / "tiny.onnx"
    written = export_model(model=toy_model, sample_input=sample_input, out_path=out)

    assert written == out.resolve()
    assert written.exists()
    assert written.stat().st_size > 0


def test_export_metadata(toy_model: TinyModel, sample_input: torch.Tensor, tmp_path: Path) -> None:
    out = tmp_path / "tiny.onnx"
    export_model(
        model=toy_model,
        sample_input=sample_input,
        out_path=out,
        metadata={"model_version": "0.0.1-test"},
    )

    loaded = onnx.load(str(out))
    props = {entry.key: entry.value for entry in loaded.metadata_props}

    assert props.get("producer_name") == "finsight-ml"
    assert "exported_at" in props
    assert props["exported_at"]  # non-empty ISO timestamp
    assert props.get("model_version") == "0.0.1-test"


def test_validate_parity_passes(
    toy_model: TinyModel, sample_input: torch.Tensor, tmp_path: Path
) -> None:
    out = tmp_path / "tiny.onnx"
    export_model(model=toy_model, sample_input=sample_input, out_path=out)

    # Mix the calibration input with a few extra random shapes to exercise the
    # parity check across more than one tensor.
    torch.manual_seed(2)
    extras = [torch.randn(1, SEQ_LEN, INPUT_DIM), torch.randn(3, SEQ_LEN, INPUT_DIM)]
    result = validate_onnx_parity(
        model=toy_model,
        onnx_path=out,
        sample_inputs=[sample_input, *extras],
    )

    assert result["n_samples"] == 3.0
    assert result["max_abs_diff"] < 1e-4
    assert "max_rel_diff" in result


def test_validate_parity_fails_on_mismatch(
    toy_model: TinyModel, sample_input: torch.Tensor, tmp_path: Path
) -> None:
    """Exporting one model and validating against a structurally different one
    must trigger the parity guard."""
    out = tmp_path / "tiny.onnx"
    export_model(model=toy_model, sample_input=sample_input, out_path=out)

    torch.manual_seed(99)
    different = DivergentModel(INPUT_DIM)

    with pytest.raises(AssertionError, match="ONNX parity check failed"):
        validate_onnx_parity(
            model=different,
            onnx_path=out,
            sample_inputs=[sample_input],
        )


def test_dynamic_batch_axis(
    toy_model: TinyModel, sample_input: torch.Tensor, tmp_path: Path
) -> None:
    out = tmp_path / "tiny.onnx"
    export_model(model=toy_model, sample_input=sample_input, out_path=out)

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    for batch in (1, 4):
        arr = np.random.RandomState(batch).randn(batch, SEQ_LEN, INPUT_DIM).astype(np.float32)
        out_arr = session.run(None, {input_name: arr})[0]
        assert out_arr.shape == (batch, 3)
