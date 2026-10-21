"""Vertex lower-star cubical PH with ordinary H0/H1 over F2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PH_FEATURE_NAMES = tuple(
    f"h{dimension}_{statistic}"
    for dimension in (0, 1)
    for statistic in (
        "count",
        "total",
        "mean",
        "std",
        "maximum",
        "entropy",
        "essential_count",
    )
)
STATISTIC_NAMES = (
    "mean",
    "std",
    "mean_absolute",
    "skewness",
    "excess_kurtosis",
) + tuple(
    f"quantile_{quantile:g}"
    for quantile in (0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1)
)
FILTRATION = "2d_grid_vertices_lower_star_sublevel_f2_nonperiodic"


@dataclass(frozen=True)
class CubicalResult:
    features: np.ndarray
    diagrams: tuple[np.ndarray, np.ndarray]


def finite_map(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError("PH requires a nonempty finite float64 2D vertex map")
    return array


def summarize_intervals(intervals: np.ndarray) -> np.ndarray:
    intervals = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    finite = intervals[np.isfinite(intervals[:, 1])]
    lifetimes = finite[:, 1] - finite[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    essential = np.isposinf(intervals[:, 1]).sum()
    if not len(lifetimes):
        return np.array([0, 0, 0, 0, 0, 0, essential], dtype=np.float64)
    proportions = lifetimes / lifetimes.sum()
    return np.array(
        [
            len(lifetimes),
            lifetimes.sum(),
            lifetimes.mean(),
            lifetimes.std(ddof=0),
            lifetimes.max(),
            -(proportions * np.log(proportions)).sum(),
            essential,
        ],
        dtype=np.float64,
    )


def cubical_features(values: np.ndarray) -> CubicalResult:
    try:
        import gudhi
    except ImportError as error:
        raise RuntimeError(
            'Install the PH extra with pip install -e ".[tda]"'
        ) from error
    array = finite_map(values)
    complex_ = gudhi.CubicalComplex(vertices=array)
    complex_.compute_persistence(homology_coeff_field=2, min_persistence=0)
    diagrams = tuple(complex_.persistence_intervals_in_dimension(d) for d in (0, 1))
    features = np.concatenate([summarize_intervals(diagram) for diagram in diagrams])
    if features[6] != 1 or features[13] != 0 or not np.isfinite(features).all():
        raise RuntimeError("invalid ordinary PH for a full finite rectangular grid")
    return CubicalResult(features, diagrams)


def distribution_statistics(values: np.ndarray) -> np.ndarray:
    array = finite_map(values).ravel()
    mean, std = array.mean(), array.std(ddof=0)
    standardized = (array - mean) / std if std else np.zeros_like(array)
    return np.concatenate(
        [
            [
                mean,
                std,
                np.abs(array).mean(),
                np.mean(standardized**3),
                np.mean(standardized**4) - 3 if std else 0,
            ],
            np.quantile(
                array, [0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1], method="linear"
            ),
        ]
    )
