"""Tiny FeatureScaler wrapping sklearn StandardScaler.

The Go inference path will reproduce the same standardization using mean and
scale parameters; persisting via joblib keeps a single source of truth.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Per-column StandardScaler with joblib persistence.

    Fit on training data only (walk-forward train slice) to avoid leakage,
    then transform train/val/test using the same parameters.
    """

    def __init__(self) -> None:
        self._scaler: StandardScaler | None = None
        self._cols: list[str] = []

    @property
    def columns(self) -> list[str]:
        return list(self._cols)

    @property
    def is_fitted(self) -> bool:
        return self._scaler is not None

    def fit_train(self, df: pd.DataFrame, cols: list[str]) -> FeatureScaler:
        """Fit on the named columns of df. Returns self."""
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"columns not present in df: {missing}")

        scaler = StandardScaler()
        scaler.fit(df[cols].to_numpy())
        self._scaler = scaler
        self._cols = list(cols)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with the configured columns standardized."""
        if self._scaler is None:
            raise RuntimeError("FeatureScaler must be fit before transform")
        missing = [c for c in self._cols if c not in df.columns]
        if missing:
            raise ValueError(f"columns not present in df: {missing}")
        out = df.copy()
        out[self._cols] = self._scaler.transform(df[self._cols].to_numpy())
        return out

    def save(self, path: Path | str) -> None:
        if self._scaler is None:
            raise RuntimeError("FeatureScaler must be fit before save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self._scaler, "cols": self._cols}, path)

    @classmethod
    def load(cls, path: Path | str) -> FeatureScaler:
        bundle = joblib.load(Path(path))
        obj = cls()
        obj._scaler = bundle["scaler"]
        obj._cols = list(bundle["cols"])
        return obj
