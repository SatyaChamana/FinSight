"""ONNX manifest sidecar.

The Go inference path needs to reproduce the exact preprocessing the Python
trainer used. Embedding every detail inside the ONNX file is fragile (binary
opaque), so we write a JSON manifest next to the model. The Go side reads it
to validate feature ordering, scaler artifact, and output column order.

Schema (v1):

    {
      "model_version": "1.4.0",
      "feature_columns": ["log_return", "rolling_vol_21", ...],
      "scaler_path": "scaler.pkl",   // or null when no scaler is used
      "input_shape": [63, 50],       // (sequence_length, feature_count)
      "output_columns": ["volatility", "var_95", "cvar_95"]
    }
"""

from __future__ import annotations

import json
from pathlib import Path


def write_manifest(
    out_path: Path | str,
    *,
    model_version: str,
    feature_columns: list[str],
    scaler_path: Path | str | None,
    input_shape: tuple[int, ...],
    output_columns: tuple[str, ...],
    symbols: list[str] | None = None,
) -> Path:
    """Write the JSON manifest next to the ONNX file.

    Args:
        out_path: Destination JSON path. Parent dirs are created if missing.
        model_version: Semver-ish string baked into Redis cache keys and the
            ONNX ``metadata_props``. Must match the version embedded in the
            exported ONNX.
        feature_columns: Exact column order the trainer fed into the model.
            The Go inference path must produce features in this order.
        scaler_path: Optional path to a serialized scaler artifact (joblib /
            pickle). ``None`` means the model was trained without scaling.
        input_shape: Static input shape excluding batch, e.g. ``(63, 50)``.
        output_columns: Named outputs in the order they appear on the last
            axis of the model output, e.g. ``("volatility", "var_95", "cvar_95")``.
        symbols: The ticker symbols the model was trained on, in the exact
            order their per-symbol feature blocks are laid out along the input
            feature axis. The Go inference path must order symbol blocks this
            way (it is insertion order, not alphabetical), so this is required
            for correct serving. ``None`` omits the field for older models.

    Returns:
        The resolved manifest path that was written.
    """
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "model_version": model_version,
        "feature_columns": list(feature_columns),
        "scaler_path": str(scaler_path) if scaler_path is not None else None,
        "input_shape": list(input_shape),
        "output_columns": list(output_columns),
    }
    if symbols is not None:
        payload["symbols"] = list(symbols)

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out_path
