"""End-to-end training pipeline.

Orchestrates:
1. yfinance download (with on-disk cache)
2. per-symbol + portfolio feature engineering
3. PortfolioRiskDataset construction
4. chronological train/val split
5. RiskPredictor training with CombinedLoss and Adam
6. checkpoint save
7. ONNX export with PyTorch / ONNX Runtime parity check
8. JSON manifest emitting feature columns and input shape so the Go
   inference path can reproduce the preprocessing exactly.

This is intentionally one file: it is the spine that wires the rest
of the package together.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data import (
    PortfolioRiskDataset,
    compute_features,
    compute_portfolio_features,
    download_ohlcv,
)
from data.features import FEATURE_COLUMNS
from data.portfolio_features import PORTFOLIO_FEATURE_COLUMNS
from export import export_model, validate_onnx_parity
from export.manifest import write_manifest
from models import CombinedLoss, RiskPredictor


@dataclass
class TrainConfig:
    """All knobs for an end-to-end run."""

    symbols: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOG", "AMZN", "META"])
    start: str = "2020-01-01"
    end: str = "2025-12-31"
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "AAPL": 0.20,
            "MSFT": 0.20,
            "GOOG": 0.20,
            "AMZN": 0.20,
            "META": 0.20,
        }
    )

    lookback_days: int = 63
    horizon_days: int = 21
    val_fraction: float = 0.15

    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 1e-3
    grad_clip_norm: float = 1.0

    hidden_dim: int = 128
    lstm_layers: int = 2
    attention_heads: int = 4

    cache_dir: Path = Path("cache")
    artifacts_dir: Path = Path("models/artifacts")
    model_version: str = "0.1.0"
    seed: int = 42


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_dataset(cfg: TrainConfig) -> PortfolioRiskDataset:
    print(f"[1/6] downloading OHLCV for {cfg.symbols} {cfg.start} -> {cfg.end}")
    ohlcv = download_ohlcv(cfg.symbols, cfg.start, cfg.end, cache_dir=cfg.cache_dir)
    print(f"      got {len(ohlcv):,} OHLCV rows across {ohlcv['symbol'].nunique()} symbols")

    print("[2/6] computing per-symbol features")
    features = compute_features(ohlcv)
    print(f"      {len(features):,} rows, columns: {list(features.columns)}")

    print("[3/6] computing portfolio features")
    portfolio_features = compute_portfolio_features(features, cfg.weights).dropna()
    print(f"      {len(portfolio_features):,} dated portfolio rows")

    print("[4/6] building PortfolioRiskDataset")
    dataset = PortfolioRiskDataset(
        features=features,
        portfolio_features=portfolio_features,
        weights=cfg.weights,
        lookback_days=cfg.lookback_days,
        horizon_days=cfg.horizon_days,
    )
    print(f"      windows: {len(dataset):,}, feature_dim: {dataset.feature_dim}")
    return dataset


def _split_train_val(dataset: PortfolioRiskDataset, val_fraction: float) -> tuple[Subset, Subset]:
    n = len(dataset)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _train_one_epoch(
    model: RiskPredictor,
    loader: DataLoader,
    loss_fn: CombinedLoss,
    optim: torch.optim.Optimizer,
    grad_clip_norm: float,
) -> dict[str, float]:
    model.train()
    totals = {"vol": 0.0, "var": 0.0, "cvar": 0.0, "total": 0.0}
    n_batches = 0
    for x, y in loader:
        optim.zero_grad()
        pred = model(x)
        loss, components = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optim.step()
        for k, v in components.items():
            totals[k] += float(v.item())
        n_batches += 1
    return {k: v / max(1, n_batches) for k, v in totals.items()}


@torch.no_grad()
def _eval(
    model: RiskPredictor,
    loader: DataLoader,
    loss_fn: CombinedLoss,
) -> dict[str, float]:
    model.eval()
    totals = {"vol": 0.0, "var": 0.0, "cvar": 0.0, "total": 0.0}
    n_batches = 0
    for x, y in loader:
        pred = model(x)
        _, components = loss_fn(pred, y)
        for k, v in components.items():
            totals[k] += float(v.item())
        n_batches += 1
    return {k: v / max(1, n_batches) for k, v in totals.items()}


def run_training(cfg: TrainConfig) -> dict[str, object]:
    """Run the full pipeline end to end. Returns a summary dict."""
    _set_seed(cfg.seed)

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset = _build_dataset(cfg)
    train_set, val_set = _split_train_val(dataset, cfg.val_fraction)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

    print(f"      train windows: {len(train_set)} | val windows: {len(val_set)}")

    print("[5/6] training")
    model = RiskPredictor(
        input_dim=dataset.feature_dim,
        hidden_dim=cfg.hidden_dim,
        lstm_layers=cfg.lstm_layers,
        attention_heads=cfg.attention_heads,
    )
    loss_fn = CombinedLoss()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    history: list[dict[str, dict[str, float]]] = []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = _train_one_epoch(model, train_loader, loss_fn, optim, cfg.grad_clip_norm)
        val_metrics = _eval(model, val_loader, loss_fn)
        history.append({"train": train_metrics, "val": val_metrics})
        print(
            f"      epoch {epoch}/{cfg.epochs}  "
            f"train total={train_metrics['total']:.4f}  "
            f"val total={val_metrics['total']:.4f}  "
            f"(vol={val_metrics['vol']:.4f} var={val_metrics['var']:.4f} cvar={val_metrics['cvar']:.4f})"
        )
    train_time = time.time() - t0
    print(f"      training took {train_time:.1f}s")

    checkpoint_path = cfg.artifacts_dir / "model.pt"
    torch.save(model, checkpoint_path)
    print(f"      saved checkpoint to {checkpoint_path}")

    print("[6/6] exporting to ONNX and verifying parity")
    onnx_path = cfg.artifacts_dir / "model.onnx"
    sample_x, _ = dataset[0]
    sample_input = sample_x.unsqueeze(0)
    export_model(
        model,
        sample_input=sample_input,
        out_path=onnx_path,
        metadata={
            "model_version": cfg.model_version,
            "feature_dim": str(dataset.feature_dim),
            "lookback_days": str(cfg.lookback_days),
        },
    )

    parity_inputs = [dataset[i][0].unsqueeze(0) for i in (0, 1, len(dataset) - 1)]
    parity = validate_onnx_parity(model, onnx_path, sample_inputs=parity_inputs, rtol=1e-3, atol=1e-4)
    print(f"      parity: {parity}")

    # Per-symbol feature columns are repeated for each symbol; the manifest
    # records the column ordering for one symbol so the Go side knows how to
    # build the flattened feature vector.
    write_manifest(
        out_path=cfg.artifacts_dir / "model.manifest.json",
        model_version=cfg.model_version,
        feature_columns=list(FEATURE_COLUMNS) + list(PORTFOLIO_FEATURE_COLUMNS),
        scaler_path=None,
        input_shape=(cfg.lookback_days, dataset.feature_dim),
        output_columns=tuple(model.output_columns),
    )
    print(f"      wrote manifest to {cfg.artifacts_dir / 'model.manifest.json'}")

    summary = {
        "symbols": cfg.symbols,
        "dataset_windows": len(dataset),
        "train_windows": len(train_set),
        "val_windows": len(val_set),
        "feature_dim": dataset.feature_dim,
        "epochs": cfg.epochs,
        "train_time_seconds": round(train_time, 2),
        "final_train_loss": history[-1]["train"]["total"],
        "final_val_loss": history[-1]["val"]["total"],
        "parity": parity,
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
        "manifest": str(cfg.artifacts_dir / "model.manifest.json"),
    }
    print("\n=== TRAINING SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_training(TrainConfig())
