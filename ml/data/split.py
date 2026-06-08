"""Walk-forward train/val/test splitting for time-series data.

Each split is an expanding-window train block followed by fixed-length val
and test blocks. Splits advance forward in time by (val + test) months so
there is no overlap between consecutive (train, val, test) bundles, and so
the model is always evaluated on data it has never seen.
"""

from __future__ import annotations

import pandas as pd
from pandas.tseries.offsets import DateOffset


def _get_date_series(df: pd.DataFrame) -> pd.Series:
    """Extract dates from either an index or a 'date' column."""
    if "date" in df.columns:
        return pd.to_datetime(df["date"])
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, index=df.index)
    raise ValueError("DataFrame must have a 'date' column or a DatetimeIndex")


def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_years: int = 4,
    val_months: int = 6,
    test_months: int = 6,
) -> list[tuple[pd.Index, pd.Index, pd.Index]]:
    """Generate n_splits expanding-window (train, val, test) index triples.

    Split k uses:
        train: [start, anchor_k]
        val:   (anchor_k, anchor_k + val_months]
        test:  (anchor_k + val_months, anchor_k + val_months + test_months]

    where anchor_0 = start + train_years, and anchor_{k+1} = anchor_k +
    (val_months + test_months). The train window grows on each iteration
    while val and test remain fixed in length.

    Args:
        df: DataFrame sorted by date (or with a DatetimeIndex).
        n_splits: Number of (train, val, test) splits to produce.
        train_years: Length in years of the initial training window.
        val_months: Length in months of the validation window.
        test_months: Length in months of the test window.

    Returns:
        List of (train_idx, val_idx, test_idx) pd.Index triples. Indices are
        the original DataFrame index values selected by the date filter.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    dates = _get_date_series(df)
    if dates.empty:
        return []

    df_sorted = df.assign(_date=dates.values).sort_values("_date")
    sorted_dates = pd.to_datetime(df_sorted["_date"])
    original_index = df_sorted.index

    start = sorted_dates.iloc[0]
    anchor = start + DateOffset(years=train_years)

    splits: list[tuple[pd.Index, pd.Index, pd.Index]] = []
    for _ in range(n_splits):
        val_end = anchor + DateOffset(months=val_months)
        test_end = val_end + DateOffset(months=test_months)

        train_mask = sorted_dates <= anchor
        val_mask = (sorted_dates > anchor) & (sorted_dates <= val_end)
        test_mask = (sorted_dates > val_end) & (sorted_dates <= test_end)

        train_idx = original_index[train_mask.values]
        val_idx = original_index[val_mask.values]
        test_idx = original_index[test_mask.values]

        splits.append((train_idx, val_idx, test_idx))

        anchor = test_end

    return splits
