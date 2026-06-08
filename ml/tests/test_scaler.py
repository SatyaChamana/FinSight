"""Tests for data.scaler.FeatureScaler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.scaler import FeatureScaler


def _make_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": rng.normal(5.0, 2.0, size=200),
            "b": rng.normal(-3.0, 0.5, size=200),
            "c": rng.uniform(0.0, 1.0, size=200),
        }
    )


def test_fit_transform_zero_mean_unit_var() -> None:
    df = _make_df()
    scaler = FeatureScaler().fit_train(df, ["a", "b"])
    out = scaler.transform(df)
    assert pytest.approx(out["a"].mean(), abs=1e-8) == 0.0
    assert pytest.approx(out["b"].mean(), abs=1e-8) == 0.0
    assert pytest.approx(out["a"].std(ddof=0), rel=1e-6) == 1.0
    assert pytest.approx(out["b"].std(ddof=0), rel=1e-6) == 1.0
    # Untouched column is unchanged.
    np.testing.assert_array_equal(out["c"].to_numpy(), df["c"].to_numpy())


def test_transform_before_fit_raises() -> None:
    df = _make_df()
    with pytest.raises(RuntimeError):
        FeatureScaler().transform(df)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    df = _make_df()
    cols = ["a", "b"]
    scaler = FeatureScaler().fit_train(df, cols)
    path = tmp_path / "scaler.joblib"
    scaler.save(path)

    loaded = FeatureScaler.load(path)
    assert loaded.is_fitted
    assert loaded.columns == cols

    original = scaler.transform(df)
    roundtripped = loaded.transform(df)
    np.testing.assert_allclose(original[cols].to_numpy(), roundtripped[cols].to_numpy())


def test_fit_train_rejects_missing_columns() -> None:
    df = _make_df()
    with pytest.raises(ValueError):
        FeatureScaler().fit_train(df, ["a", "missing"])
