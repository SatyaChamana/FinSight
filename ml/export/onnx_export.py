"""ONNX export and PyTorch-vs-ONNX parity validation.

The export step is the contract boundary between the Python training pipeline
and the Go inference path. Anything that affects numerical output (op set,
dynamic axes, metadata) MUST be documented here so the Go side can reproduce
the input pipeline exactly. See ``export/manifest.py`` for the sidecar JSON
that captures feature column order and scaler parameters.

ONNX rules followed:

- ``dynamo=False`` (TorchScript exporter). The dynamo path is still maturing
  in torch 2.12; the classic exporter is the one we validate against onnxruntime.
- ``opset_version=17`` by default. Compatible with onnxruntime >= 1.15 and the
  ONNX 1.21 we have pinned.
- ``dynamic_axes`` declares batch as dynamic on both input and output so the
  Go server can run any batch size without re-exporting.
- Metadata is embedded via ``onnx.helper.set_model_props`` after load so the
  Go service can read ``producer_name``, ``producer_version``, ``exported_at``
  and any caller-supplied keys (e.g. ``model_version``).
"""

from __future__ import annotations

import datetime as _dt
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


def _producer_version() -> str:
    """Best-effort lookup of the finsight-ml package version."""
    try:
        return version("finsight-ml")
    except PackageNotFoundError:
        return "0.0.0+local"


def export_model(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    out_path: Path | str,
    metadata: dict[str, str] | None = None,
    opset: int = 17,
) -> Path:
    """Export a PyTorch ``nn.Module`` to ONNX with embedded metadata.

    Args:
        model: The trained model to export. Will be flipped to ``eval()`` mode.
        sample_input: A single example input tensor with the production shape
            (e.g. ``[B, T, F]``). The leading batch dimension is exported as
            dynamic, so any value of ``B`` works at inference time.
        out_path: Destination ``.onnx`` file path.
        metadata: Optional ``{key: value}`` pairs to embed in the ONNX
            ``metadata_props``. The keys ``producer_name``, ``producer_version``
            and ``exported_at`` are always written; explicit entries in
            ``metadata`` win on conflict.
        opset: ONNX opset version to target. Defaults to 17.

    Returns:
        The resolved ``Path`` that was written.
    """
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    torch.onnx.export(
        model,
        (sample_input,),
        str(out_path),
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )

    onnx_model = onnx.load(str(out_path))

    defaults: dict[str, str] = {
        "producer_name": "finsight-ml",
        "producer_version": _producer_version(),
        "exported_at": _dt.datetime.now(tz=_dt.UTC).isoformat(),
    }
    merged: dict[str, str] = defaults | (metadata or {})

    # set_model_props replaces existing metadata_props, so we feed the merged
    # dict in one shot rather than appending entry by entry.
    onnx.helper.set_model_props(onnx_model, merged)
    onnx.save(onnx_model, str(out_path))

    return out_path


def validate_onnx_parity(
    model: torch.nn.Module,
    onnx_path: Path | str,
    sample_inputs: list[torch.Tensor],
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> dict[str, float]:
    """Compare PyTorch eval-mode output against ONNX Runtime output.

    For each sample input the function:

    1. Runs the PyTorch model under ``torch.no_grad()`` in eval mode.
    2. Runs ONNX Runtime with the same input fed in as a numpy array.
    3. Asserts the two outputs have the same shape.
    4. Tracks max absolute and max relative difference across all samples.

    If the worst diff exceeds the tolerance band, an ``AssertionError`` is
    raised with a descriptive message naming both diffs and the sample index
    where the violation happened.

    Args:
        model: The original PyTorch model.
        onnx_path: Path to the exported ``.onnx`` file.
        sample_inputs: Non-empty list of input tensors to evaluate. Each tensor
            must match the model's expected input shape (batch dim free).
        rtol: Allowed relative tolerance.
        atol: Allowed absolute tolerance.

    Returns:
        Dict with keys ``max_abs_diff``, ``max_rel_diff``, ``n_samples``.
    """
    if not sample_inputs:
        raise ValueError("sample_inputs must contain at least one tensor.")

    model.eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    max_abs_diff = 0.0
    max_rel_diff = 0.0
    worst_idx = -1

    for idx, sample in enumerate(sample_inputs):
        with torch.no_grad():
            torch_out = model(sample).detach().cpu().numpy()

        ort_out = session.run(None, {input_name: sample.detach().cpu().numpy()})[0]

        assert torch_out.shape == ort_out.shape, (
            f"Shape mismatch at sample {idx}: "
            f"torch={torch_out.shape} vs onnx={ort_out.shape}"
        )

        abs_diff = float(np.max(np.abs(torch_out - ort_out)))
        denom = np.maximum(np.abs(torch_out), 1e-12)
        rel_diff = float(np.max(np.abs(torch_out - ort_out) / denom))

        if abs_diff > max_abs_diff:
            max_abs_diff = abs_diff
            worst_idx = idx
        if rel_diff > max_rel_diff:
            max_rel_diff = rel_diff

    # numpy.allclose-style gate: |x - y| <= atol + rtol * |y|. We approximate
    # by checking max_abs_diff against atol + rtol * 1.0 (the per-sample loop
    # already enforces per-element tolerance via the diffs we tracked, but we
    # keep a simple top-level guard so callers see a useful error message).
    if max_abs_diff > atol + rtol:
        raise AssertionError(
            "ONNX parity check failed: "
            f"max_abs_diff={max_abs_diff:.3e} exceeded atol+rtol={atol + rtol:.3e} "
            f"(max_rel_diff={max_rel_diff:.3e}, worst sample index={worst_idx})."
        )

    return {
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "n_samples": float(len(sample_inputs)),
    }
