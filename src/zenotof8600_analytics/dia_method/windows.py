"""Designing a variable-width DIA isolation-window scheme."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import _oadia

__all__ = [
    'WindowScheme',
    'design_windows',
    'load_precursor_mz',
    'window_occupancy',
    'evaluate_windows',
]

_MZ_COLUMNS = ('mz', 'm/z', 'Precursor.Mz', 'precursor_mz', 'mz_values', 'MZ')


def load_precursor_mz(
    source: str | os.PathLike | pd.DataFrame | pd.Series | Iterable[float],
    column: str | None = None,
) -> list[float]:
    """Return precursor m/z values as a plain list of floats."""
    if isinstance(source, (str, os.PathLike)):
        path = str(source)
        frame = (pd.read_parquet(path) if path.lower().endswith('.parquet')
                 else pd.read_csv(path))
        source = frame

    if isinstance(source, pd.DataFrame):
        if column is None:
            for candidate in _MZ_COLUMNS:
                if candidate in source.columns:
                    column = candidate
                    break
            else:
                raise KeyError(
                    'No precursor m/z column found. Tried '
                    f'{list(_MZ_COLUMNS)}; the table has '
                    f'{list(source.columns)[:12]}. Pass column=... explicitly.'
                )
        values = source[column]
    else:
        values = pd.Series(list(source), dtype='float64')

    values = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
    if values.empty:
        raise ValueError('No usable precursor m/z values found.')
    return values.astype(float).tolist()


@dataclass(frozen=True)
class WindowScheme:
    """A contiguous set of DIA isolation windows, plus how it was built."""

    bins: list[list[float]]
    mz_range: tuple[float, float]
    n_windows_requested: int
    window_type: str = 'dynamic'
    min_width: float = 1.0
    max_width: float = 100.0
    forbidden_zones: bool = True
    phospho: bool = False
    n_precursors: int | None = field(default=None, compare=False)

    def __len__(self) -> int:
        return len(self.bins)

    @property
    def n_windows(self) -> int:
        """Windows actually produced, which can differ from the number asked for: the min-width merge step can return fewer."""
        return len(self.bins)

    @property
    def starts(self) -> np.ndarray:
        return np.asarray([b[0] for b in self.bins], dtype=float)

    @property
    def ends(self) -> np.ndarray:
        return np.asarray([b[1] for b in self.bins], dtype=float)

    @property
    def widths(self) -> np.ndarray:
        return self.ends - self.starts

    @property
    def centers(self) -> np.ndarray:
        return (self.starts + self.ends) / 2

    @property
    def covered_range(self) -> tuple[float, float]:
        """The m/z actually covered."""
        return (float(self.starts[0]), float(self.ends[-1]))

    @property
    def table(self) -> pd.DataFrame:
        """One row per window -- the source-data form of the scheme."""
        return pd.DataFrame({
            'window': np.arange(1, self.n_windows + 1),
            'window_start': self.starts,
            'window_end': self.ends,
            'window_center': self.centers,
            'window_width': self.widths,
        })

    def validate(self, tol: float = 1e-6) -> 'WindowScheme':
        """Raise unless the windows form a clean partition."""
        if self.n_windows == 0:
            raise ValueError('Scheme has no windows.')
        if np.any(self.widths <= 0):
            bad = int(np.argmin(self.widths)) + 1
            raise ValueError(f'Window {bad} has non-positive width.')
        gaps = self.starts[1:] - self.ends[:-1]
        if np.any(np.abs(gaps) > tol):
            worst = int(np.argmax(np.abs(gaps)))
            raise ValueError(
                f'Windows {worst + 1} and {worst + 2} are not contiguous '
                f'({gaps[worst]:+.6f} Th gap/overlap).'
            )
        return self

    def on_forbidden_zones(self, tol: float = 1e-6) -> bool:
        """True if every boundary sits on a forbidden zone."""
        edges = np.concatenate([self.starts, self.ends[-1:]])
        snapped = [_oadia.find_closest_forbidden_zone(v, self.phospho) for v in edges]
        return bool(np.allclose(snapped, edges, atol=tol))

    def summary(self) -> str:
        lo, hi = self.covered_range
        w = self.widths
        return (
            f'{self.n_windows} {self.window_type} windows, '
            f'{lo:.3f}-{hi:.3f} m/z; width min/median/max '
            f'{w.min():.2f}/{np.median(w):.2f}/{w.max():.2f} Th'
            + ('; boundaries on forbidden zones' if self.forbidden_zones else '')
        )


def design_windows(
    mz_values: Sequence[float] | pd.Series | np.ndarray,
    mz_range: tuple[float, float],
    n_windows: int,
    window_type: str = 'dynamic',
    min_width: float = 1.0,
    max_width: float = 100.0,
    forbidden_zones: bool = True,
    phospho: bool = False,
) -> WindowScheme:
    """Design an isolation-window scheme over a precursor m/z distribution."""
    if window_type not in ('fixed', 'dynamic'):
        raise ValueError(
            f"window_type {window_type!r} not supported; use 'fixed' or 'dynamic'."
        )
    lo, hi = float(mz_range[0]), float(mz_range[1])
    if hi <= lo:
        raise ValueError(f'mz_range must be increasing, got ({lo}, {hi}).')
    if n_windows < 1:
        raise ValueError(f'n_windows must be >= 1, got {n_windows}.')

    values = load_precursor_mz(mz_values)
    in_range = sum(1 for v in values if lo <= v <= hi)
    if window_type == 'dynamic' and in_range == 0:
        raise ValueError(
            f'No precursors fall inside m/z {lo}-{hi}, so a dynamic scheme has '
            'nothing to balance. Widen the range or use window_type="fixed".'
        )

    if window_type == 'dynamic':
        bins = _oadia.create_variable_bins(values, (lo, hi), n_windows,
                                           min_width, max_width)
    else:
        bins = _oadia.get_initial_equal_width_bins((lo, hi), n_windows)

    if forbidden_zones:
        bins = _oadia.adjust_bin_boundaries(bins, phospho_enriched=phospho)

    scheme = WindowScheme(
        bins=[[float(a), float(b)] for a, b in bins],
        mz_range=(lo, hi),
        n_windows_requested=n_windows,
        window_type=window_type,
        min_width=min_width,
        max_width=max_width,
        forbidden_zones=forbidden_zones,
        phospho=phospho,
        n_precursors=in_range,
    )
    return scheme.validate()


def window_occupancy(
    scheme: WindowScheme,
    mz_values: Sequence[float] | pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Per-window precursor counts for a given distribution."""
    values = np.sort(np.asarray(load_precursor_mz(mz_values), dtype=float))
    edges = np.concatenate([scheme.starts, scheme.ends[-1:]])
    counts = np.diff(np.searchsorted(values, edges, side='left'))
    counts[-1] += int(np.sum(values == edges[-1]))

    table = scheme.table
    table['n_precursors'] = counts
    table['precursor_density'] = counts / table['window_width']
    return table


def evaluate_windows(
    scheme: WindowScheme,
    mz_values: Sequence[float] | pd.Series | np.ndarray,
) -> dict:
    """Summary statistics for a scheme against a precursor distribution."""
    occupancy = window_occupancy(scheme, mz_values)
    counts = occupancy['n_precursors'].to_numpy()
    widths = occupancy['window_width'].to_numpy()

    values = np.asarray(load_precursor_mz(mz_values), dtype=float)
    lo, hi = scheme.covered_range
    inside = int(np.sum((values >= lo) & (values <= hi)))

    return {
        'n_windows': scheme.n_windows,
        'window_type': scheme.window_type,
        'mz_range': scheme.mz_range,
        'covered_range': scheme.covered_range,
        'width_min': float(widths.min()),
        'width_median': float(np.median(widths)),
        'width_mean': float(widths.mean()),
        'width_max': float(widths.max()),
        'count_min': int(counts.min()),
        'count_median': float(np.median(counts)),
        'count_mean': float(counts.mean()),
        'count_max': int(counts.max()),
        'balance': float(counts.max() / counts.min()) if counts.min() else np.inf,
        'n_precursors_total': int(values.size),
        'n_precursors_covered': inside,
        'coverage_pct': round(100 * inside / values.size, 2) if values.size else 0.0,
        'on_forbidden_zones': scheme.on_forbidden_zones(),
    }
