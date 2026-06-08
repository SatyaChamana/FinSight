"""Tests for data.synthetic.generate_stress_scenario."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.download import OHLCV_COLUMNS
from data.synthetic import SCENARIOS, generate_stress_scenario

SCHEMA_COLUMNS = set(OHLCV_COLUMNS)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_preserves_schema(
    synthetic_ohlcv_three_symbols: pd.DataFrame, scenario: str
) -> None:
    out = generate_stress_scenario(synthetic_ohlcv_three_symbols, scenario, severity=1.0)
    assert set(out.columns) == SCHEMA_COLUMNS
    # Same row count per symbol as the input (we perturb, we do not drop rows).
    base_counts = synthetic_ohlcv_three_symbols.groupby("symbol").size()
    out_counts = out.groupby("symbol").size()
    assert base_counts.equals(out_counts)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_actually_changes_data(
    synthetic_ohlcv_three_symbols: pd.DataFrame, scenario: str
) -> None:
    out = generate_stress_scenario(synthetic_ohlcv_three_symbols, scenario, severity=1.0)
    base = synthetic_ohlcv_three_symbols.sort_values(["symbol", "date"]).reset_index(drop=True)
    out_sorted = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    diff = (out_sorted[["open", "high", "low", "close", "adj_close"]].to_numpy()
            - base[["open", "high", "low", "close", "adj_close"]].to_numpy())
    assert np.any(np.abs(diff) > 1e-8), f"scenario {scenario} did not perturb the data"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_severity_zero_returns_base(
    synthetic_ohlcv_three_symbols: pd.DataFrame, scenario: str
) -> None:
    out = generate_stress_scenario(synthetic_ohlcv_three_symbols, scenario, severity=0.0)
    base = synthetic_ohlcv_three_symbols.copy().reset_index(drop=True)
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True),
        base.reset_index(drop=True),
        check_dtype=False,
    )


def test_unknown_scenario_raises(synthetic_ohlcv_three_symbols: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        generate_stress_scenario(synthetic_ohlcv_three_symbols, "no_such_scenario")
