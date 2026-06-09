"""Intensive, generalized training on a large symbol universe.

Differences from training/train.py (the single-portfolio teaching run):

1. Vast data: downloads a large universe over a long horizon and samples many
   random 5-asset portfolios (random weights), so the model sees a huge,
   diverse set of windows instead of one portfolio's history.
2. Symbol-agnostic: each sampled portfolio lays its per-symbol blocks in a
   RANDOM column order, so the model cannot memorize a fixed symbol-per-column
   mapping and instead generalizes to any 5-asset portfolio. The exported
   manifest therefore omits the `symbols` field (the Go feature builder treats
   a symbol-less manifest as "generalized" and orders the portfolio's holdings
   deterministically).
3. Intensive optimization: MPS (Apple GPU) when available, larger batches,
   many epochs with early stopping, ReduceLROnPlateau, gradient clipping, and
   an explicit CVaR>=VaR ordering penalty so the three heads stay internally
   consistent (the original loss only trained CVaR on tail rows).

Reuses the exact feature pipeline (PortfolioRiskDataset, compute_features,
compute_portfolio_features) so Go-side parity is preserved.

Run from ml/:  PYTHONPATH=. .venv/bin/python -m training.train_large
Quick pipeline check (tiny run to /tmp):  add --smoke, or set SMOKE=1.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

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

# A diversified, liquid universe with long history. Per-portfolio date alignment
# means later IPOs (e.g. META 2012) simply yield shorter ranges for portfolios
# that include them; they do not truncate the whole dataset.
UNIVERSE: list[str] = [
    # mega-cap tech / semis
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "ADBE", "CRM", "ORCL",
    "CSCO", "INTC", "AMD", "QCOM", "TXN",
    # financials
    "JPM", "BAC", "WFC", "GS", "MS", "C",
    # health care
    "JNJ", "UNH", "PFE", "MRK", "ABT", "TMO",
    # consumer
    "KO", "PEP", "PG", "WMT", "COST", "MCD", "NKE", "DIS", "HD",
    # energy / industrial
    "XOM", "CVX", "CAT", "BA", "HON",
]


@dataclass
class LargeTrainConfig:
    universe: list[str] = field(default_factory=lambda: list(UNIVERSE))
    start: str = "2010-01-01"
    end: str = "2025-12-31"

    portfolio_size: int = 5
    n_portfolios: int = 140
    per_portfolio_cap: int = 1400  # max windows sampled from each portfolio
    max_windows: int = 160_000     # global cap (subsampled if exceeded)

    lookback_days: int = 63
    horizon_days: int = 21
    val_cutoff: str = "2023-06-01"  # windows as-of >= this date are validation

    batch_size: int = 256
    epochs: int = 80
    patience: int = 8
    select_loss_round: int = 4  # val-loss buckets for the selection tie-break
    learning_rate: float = 1e-3
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-5
    w_order: float = 0.5  # CVaR>=VaR ordering penalty weight

    hidden_dim: int = 128
    lstm_layers: int = 2
    attention_heads: int = 4

    cache_dir: Path = Path("cache")
    artifacts_dir: Path = Path("models/artifacts")
    model_version: str = "1.0.0"
    seed: int = 42


@torch.no_grad()
def _evaluate(
    model: RiskPredictor,
    X: torch.Tensor,
    Y: torch.Tensor,
    device: torch.device,
    batch_size: int,
    loss_fn: CombinedLoss,
    w_order: float,
) -> dict[str, float]:
    """Batched validation so a large val set does not allocate one giant LSTM
    buffer. Returns mean loss components, total (incl. order penalty), and the
    fraction of rows where cvar_pred < var_pred."""
    model.eval()
    n = len(X)
    acc = {"vol": 0.0, "var": 0.0, "cvar": 0.0, "base": 0.0, "order": 0.0}
    viol = 0
    cnt = 0
    for s in range(0, n, batch_size):
        xb = X[s : s + batch_size].to(device)
        yb = Y[s : s + batch_size].to(device)
        pred = model(xb)
        _, c = loss_fn(pred, yb)
        bs = len(xb)
        acc["vol"] += float(c["vol"]) * bs
        acc["var"] += float(c["var"]) * bs
        acc["cvar"] += float(c["cvar"]) * bs
        acc["base"] += float(c["total"]) * bs
        acc["order"] += float(torch.relu(pred[:, 1] - pred[:, 2]).sum().item())
        viol += int((pred[:, 2] < pred[:, 1]).sum().item())
        cnt += bs
    cnt = max(1, cnt)
    order_mean = acc["order"] / cnt
    return {
        "vol": acc["vol"] / cnt,
        "var": acc["var"] / cnt,
        "cvar": acc["cvar"] / cnt,
        "total": acc["base"] / cnt + w_order * order_mean,
        "cvar_lt_var_frac": viol / cnt,
    }


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sample_portfolio(rng: np.random.Generator, universe: list[str], size: int) -> dict[str, float]:
    syms = list(rng.choice(universe, size=size, replace=False))
    rng.shuffle(syms)  # random column order -> symbol-agnostic model
    w = rng.dirichlet(np.ones(size))
    w = w / w.sum()  # exact simplex within float eps
    return {s: float(wi) for s, wi in zip(syms, w, strict=True)}


def _build_windows(cfg: LargeTrainConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns X [N,63,49], Y [N,3], asof_ts [N] (int64 ns) across many sampled
    portfolios."""
    print(f"[1/4] downloading {len(cfg.universe)} symbols {cfg.start}->{cfg.end}")
    ohlcv = download_ohlcv(cfg.universe, cfg.start, cfg.end, cache_dir=cfg.cache_dir)
    print(f"      {len(ohlcv):,} rows, {ohlcv['symbol'].nunique()} symbols")

    print("[2/4] computing per-symbol features (once)")
    features_all = compute_features(ohlcv)
    present = sorted(features_all["symbol"].unique())
    print(f"      features for {len(present)} symbols, {len(features_all):,} rows")

    rng = np.random.default_rng(cfg.seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []

    print(f"[3/4] sampling {cfg.n_portfolios} portfolios")
    built = 0
    attempts = 0
    while built < cfg.n_portfolios and attempts < cfg.n_portfolios * 4:
        attempts += 1
        weights = _sample_portfolio(rng, present, cfg.portfolio_size)
        syms = list(weights.keys())
        feat_sub = features_all[features_all["symbol"].isin(syms)]
        try:
            pf = compute_portfolio_features(feat_sub, weights).dropna()
            ds = PortfolioRiskDataset(
                features=feat_sub,
                portfolio_features=pf,
                weights=weights,
                lookback_days=cfg.lookback_days,
                horizon_days=cfg.horizon_days,
            )
        except ValueError:
            continue  # not enough aligned history for this combo
        n = len(ds)
        if n < 50:
            continue

        idxs = np.arange(n)
        if n > cfg.per_portfolio_cap:
            idxs = rng.choice(idxs, size=cfg.per_portfolio_cap, replace=False)
        dates = ds.dates
        for i in idxs:
            x, y = ds[int(i)]
            xs.append(x.numpy())
            ys.append(y.numpy())
            ts.append(np.int64(dates[int(i) + cfg.lookback_days - 1].value))
        built += 1
        if built % 20 == 0:
            print(f"      {built}/{cfg.n_portfolios} portfolios, {len(xs):,} windows so far")

    X = np.stack(xs).astype(np.float32)
    Y = np.stack(ys).astype(np.float32)
    T = np.asarray(ts, dtype=np.int64)
    print(f"      total windows: {len(X):,}")

    if len(X) > cfg.max_windows:
        sel = rng.choice(len(X), size=cfg.max_windows, replace=False)
        X, Y, T = X[sel], Y[sel], T[sel]
        print(f"      subsampled to {len(X):,} (max_windows)")
    return X, Y, T


def _train(cfg: LargeTrainConfig) -> dict[str, object]:
    _set_seed(cfg.seed)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    device = _device()
    print(f"device: {device}")

    X, Y, T = _build_windows(cfg)

    cutoff = np.int64(np.datetime64(cfg.val_cutoff, "ns").astype("int64"))
    val_mask = cutoff <= T
    train_mask = ~val_mask
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        # Fallback for tiny smoke runs: last 15% chronologically as validation.
        order = np.argsort(T)
        n_val = max(1, int(0.15 * len(T)))
        val_idx = order[-n_val:]
        val_mask = np.zeros(len(T), dtype=bool)
        val_mask[val_idx] = True
        train_mask = ~val_mask

    # Keep tensors on CPU; batches are moved to the device on demand so neither
    # the train nor the val pass allocates one oversized LSTM buffer.
    Xtr = torch.from_numpy(X[train_mask])
    Ytr = torch.from_numpy(Y[train_mask])
    Xva = torch.from_numpy(X[val_mask])
    Yva = torch.from_numpy(Y[val_mask])
    print(f"      train windows: {len(Xtr):,} | val windows: {len(Xva):,}")

    model = RiskPredictor(
        input_dim=X.shape[2],
        hidden_dim=cfg.hidden_dim,
        lstm_layers=cfg.lstm_layers,
        attention_heads=cfg.attention_heads,
    ).to(device)
    loss_fn = CombinedLoss()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=3)

    n = len(Xtr)
    # Selection key is (bucketed val loss, cvar<var fraction): among epochs whose
    # val loss rounds to the same bucket (a near-tie), the one with the fewest
    # CVaR<VaR violations wins. A clearly lower bucket still wins outright. This
    # avoids exporting a marginally-lower-loss model that has worse head ordering.
    best_key: tuple[float, float] | None = None
    best_val = float("inf")
    best_viol = 1.0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    history: list[dict[str, float]] = []

    print(f"[4/4] training up to {cfg.epochs} epochs (early stop patience {cfg.patience})")
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        running = 0.0
        nb = 0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            xb = Xtr[idx].to(device)
            yb = Ytr[idx].to(device)
            optim.zero_grad()
            pred = model(xb)
            loss, _ = loss_fn(pred, yb)
            order_pen = torch.relu(pred[:, 1] - pred[:, 2]).mean()  # var - cvar > 0 penalized
            loss = loss + cfg.w_order * order_pen
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optim.step()
            running += float(loss.item())
            nb += 1
        train_loss = running / max(1, nb)

        ev = _evaluate(model, Xva, Yva, device, cfg.batch_size, loss_fn, cfg.w_order)
        val_total = ev["total"]
        viol = ev["cvar_lt_var_frac"]
        sched.step(val_total)
        history.append({"epoch": epoch, "train": train_loss, "val": val_total, "cvar_lt_var_frac": viol})
        print(
            f"      epoch {epoch:3d}  train={train_loss:.5f}  val={val_total:.5f}  "
            f"(vol={ev['vol']:.4f} var={ev['var']:.4f} cvar={ev['cvar']:.4f} "
            f"cvar<var={viol*100:.1f}%)"
        )

        key = (round(val_total, cfg.select_loss_round), viol)
        if best_key is None or key < best_key:
            best_key = key
            best_val = val_total
            best_viol = viol
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(
                    f"      early stop at epoch {epoch} "
                    f"(selected val {best_val:.5f}, cvar<var {best_viol*100:.1f}%)"
                )
                break
    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to("cpu").eval()

    print("exporting best model to ONNX")
    onnx_path = cfg.artifacts_dir / "model.onnx"
    sample_input = torch.from_numpy(X[:1])
    export_model(
        model,
        sample_input=sample_input,
        out_path=onnx_path,
        metadata={
            "model_version": cfg.model_version,
            "feature_dim": str(X.shape[2]),
            "lookback_days": str(cfg.lookback_days),
        },
    )
    parity_inputs = [torch.from_numpy(X[i : i + 1]) for i in (0, 1, len(X) - 1)]
    parity = validate_onnx_parity(model, onnx_path, sample_inputs=parity_inputs, rtol=1e-3, atol=1e-4)
    torch.save(model, cfg.artifacts_dir / "model.pt")

    write_manifest(
        out_path=cfg.artifacts_dir / "model.manifest.json",
        model_version=cfg.model_version,
        feature_columns=list(FEATURE_COLUMNS) + list(PORTFOLIO_FEATURE_COLUMNS),
        scaler_path=None,
        input_shape=(cfg.lookback_days, X.shape[2]),
        output_columns=tuple(model.output_columns),
        symbols=None,  # generalized model: serves any 5-asset portfolio
    )

    summary = {
        "model_version": cfg.model_version,
        "universe_size": len(cfg.universe),
        "windows_total": int(len(X)),
        "train_windows": int(train_mask.sum()),
        "val_windows": int(val_mask.sum()),
        "epochs_run": len(history),
        "best_val": round(best_val, 6),
        "selected_cvar_lt_var_frac": round(best_viol, 6),
        "final_epoch_cvar_lt_var_frac": history[-1]["cvar_lt_var_frac"],
        "train_time_seconds": round(train_time, 1),
        "parity": parity,
        "device": str(device),
        "generalized": True,
    }
    print("\n=== INTENSIVE TRAINING SUMMARY ===")
    print(json.dumps(summary, indent=2))
    (cfg.artifacts_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _smoke(cfg: LargeTrainConfig) -> LargeTrainConfig:
    cfg.universe = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "JPM", "KO"]
    cfg.start = "2018-01-01"
    cfg.n_portfolios = 4
    cfg.per_portfolio_cap = 150
    cfg.max_windows = 600
    cfg.epochs = 2
    cfg.patience = 2
    cfg.artifacts_dir = Path("/tmp/finsight_smoke_artifacts")  # do not clobber the real model
    return cfg


def _smoke_requested(flag: bool) -> bool:
    """Smoke mode is on via the --smoke flag OR a truthy SMOKE env var
    (SMOKE=1/true/yes), so both invocation styles work."""
    if flag:
        return True
    return os.getenv("SMOKE", "").strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to validate the pipeline")
    args = ap.parse_args()
    config = LargeTrainConfig()
    if _smoke_requested(args.smoke):
        print("SMOKE mode: tiny fast run")
        config = _smoke(config)
    _train(config)
