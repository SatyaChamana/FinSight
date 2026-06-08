# Python ML Pipeline -- ml/

## Developer Reminder

The developer knows Python well. No need to explain Python basics, but do explain ML/ONNX concepts clearly.

## Directory Structure

- `data/` -- Data loading, preprocessing, feature engineering scripts
- `models/` -- Model architecture definitions (LSTM, Transformer)
- `training/` -- Training loops, hyperparameter configs, experiment tracking
- `export/` -- ONNX export scripts, model validation post-export
- `notebooks/` -- Exploration and prototyping Jupyter notebooks

## Conventions

- All function signatures have type hints.
- Use pandas and numpy for data processing.
- PyTorch for model training.
- ONNX for model export (must match Go inference preprocessing exactly).
- Tests use pytest. Run with `python -m pytest` from `ml/` directory.
- Lint with ruff, format with black.
- Pin dependencies in `requirements.txt` with exact versions.

## ONNX Export Rules

- Every preprocessing step in the training pipeline must be documented and replicated identically in the Go inference path (`internal/model/`).
- Export script must validate ONNX model output matches PyTorch output within tolerance before saving.
- Model metadata (version, input shape, feature names) must be embedded in ONNX model properties.
