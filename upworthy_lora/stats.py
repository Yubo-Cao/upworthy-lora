"""Bootstrap confidence intervals over per-group scores."""

from __future__ import annotations

import numpy as np


def macro_mean_ci(
    values: list[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of per-group scores with a percentile bootstrap CI.

    The group is the resampling unit. Headlines within a group are not
    independent (they competed against each other), so bootstrapping over
    headlines would understate the interval.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return (
        float(arr.mean()),
        float(np.quantile(draws, alpha / 2)),
        float(np.quantile(draws, 1 - alpha / 2)),
    )
