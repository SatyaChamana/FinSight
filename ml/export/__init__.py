"""FinSight ML ONNX export utilities.

Public surface used by ``cli.py`` and the integration training step:

- ``export_model``: PyTorch ``nn.Module`` to ONNX with embedded metadata.
- ``validate_onnx_parity``: PyTorch vs ONNX Runtime output equivalence check.
- ``write_manifest``: writes a JSON sidecar describing preprocessing (feature
  columns, scaler params path) so the Go inference path can reproduce inputs
  exactly.
"""

from export.manifest import write_manifest
from export.onnx_export import export_model, validate_onnx_parity

__all__ = [
    "export_model",
    "validate_onnx_parity",
    "write_manifest",
]
