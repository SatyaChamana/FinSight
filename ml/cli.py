"""FinSight ML command-line interface.

A small typer app that wires the three core workflows together:

1. ``download`` pulls OHLCV from yfinance into a local parquet cache.
2. ``train`` runs the LSTM + attention training loop (placeholder for now;
   the real loop is wired up in a later integration step).
3. ``export`` converts a PyTorch checkpoint into an ONNX file, embedding the
   metadata Go inference needs.

The entry point is exposed as the ``finsight-ml`` console script via
``pyproject.toml`` -> ``[project.scripts]``.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, help="FinSight ML pipeline CLI.")


@app.command()
def download(
    symbols: str = typer.Option(
        ...,
        "--symbols",
        help="Comma-separated ticker symbols, e.g. 'AAPL,MSFT,GOOG'.",
    ),
    start: str = typer.Option(..., "--start", help="ISO start date (inclusive)."),
    end: str = typer.Option(..., "--end", help="ISO end date (exclusive, yfinance convention)."),
    out: Path = typer.Option(
        Path("./cache"),
        "--out",
        help="Cache directory for per-symbol parquet files.",
    ),
) -> None:
    """Download OHLCV history for ``--symbols`` between ``--start`` and ``--end``."""
    # Imported lazily so the CLI's --help works even if data/ deps are missing.
    from data import download_ohlcv

    out.mkdir(parents=True, exist_ok=True)
    symbol_list: list[str] = [s.strip() for s in symbols.split(",") if s.strip()]
    df = download_ohlcv(symbols=symbol_list, start=start, end=end, cache_dir=out)
    typer.echo(f"Downloaded {len(df)} rows for {len(symbol_list)} symbols into {out}.")


@app.command()
def train(
    symbols: str = typer.Option(
        "AAPL,MSFT,GOOG,AMZN,META",
        "--symbols",
        help="Comma-separated tickers for the demo portfolio (equal weights).",
    ),
    start: str = typer.Option("2020-01-01", "--start"),
    end: str = typer.Option("2025-12-31", "--end"),
    epochs: int = typer.Option(3, "--epochs"),
    batch_size: int = typer.Option(32, "--batch-size"),
    artifacts_dir: Path = typer.Option(
        Path("./models/artifacts"),
        "--artifacts-dir",
        help="Where to write model.pt, model.onnx, and model.manifest.json.",
    ),
) -> None:
    """Train the LSTM + attention model end-to-end.

    Downloads OHLCV, engineers features, fits the model, exports to ONNX, and
    validates PyTorch / ONNX parity. Defaults to a five-ticker equal-weight
    portfolio so a fresh checkout can sanity-check the pipeline in under a
    minute on CPU.
    """
    from training.train import TrainConfig, run_training

    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if len(symbol_list) == 0:
        raise typer.BadParameter("--symbols cannot be empty")
    equal_weight = round(1.0 / len(symbol_list), 6)
    weights = dict.fromkeys(symbol_list, equal_weight)

    cfg = TrainConfig(
        symbols=symbol_list,
        start=start,
        end=end,
        weights=weights,
        epochs=epochs,
        batch_size=batch_size,
        artifacts_dir=artifacts_dir,
    )
    summary = run_training(cfg)
    typer.echo(
        f"Done. checkpoint={summary['checkpoint']} onnx={summary['onnx']} "
        f"val_loss={summary['final_val_loss']:.4f}"
    )


@app.command()
def export(
    checkpoint: Path = typer.Option(
        Path("./models/artifacts/model.pt"),
        "--checkpoint",
        help="Path to the PyTorch checkpoint to export.",
    ),
    out: Path = typer.Option(
        Path("./models/artifacts/model.onnx"),
        "--out",
        help="Where to write the resulting ONNX file.",
    ),
) -> None:
    """Export a trained checkpoint to ONNX, embedding producer/version metadata."""
    # Imported lazily so unrelated commands don't pay the torch import cost.
    import torch

    from export import export_model

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, torch.nn.Module):
        model = state
    else:
        # Checkpoints that bundle (model, optimizer, ...) are handled by the
        # integration step that knows the concrete architecture; for now we
        # require a serialized nn.Module.
        raise typer.BadParameter(
            "Checkpoint must be a torch.nn.Module instance until the training "
            "integration lands. Got a state-dict-style payload instead."
        )

    # The integration step will replace this with the real feature tensor shape.
    sample_input = torch.zeros(1, 63, 50)
    written = export_model(model=model, sample_input=sample_input, out_path=out)
    typer.echo(f"Wrote ONNX model to {written}.")


if __name__ == "__main__":
    app()
