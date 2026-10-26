"""Scan selection for TARGETED acquisitions (MRM-HR / PRM)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .extraction import _SpectrumView, extract_xic, resolve_mz_tol

__all__ = [
    'target_scans',
    'targeted_xic',
    'is_targeted',
    'assert_targets_resolved',
]

DEFAULT_MATCH_TH = 0.05


def is_targeted(ms_data, *, ms_level=2, max_windows=200):
    """Does this file look like a targeted acquisition rather than DIA?"""
    key = (ms_level, max_windows, len(ms_data.spectrum_df))
    memo = getattr(ms_data, '_is_targeted_memo', None)
    if memo is not None and key in memo:
        return memo[key]
    result = _is_targeted(ms_data.spectrum_df, ms_level, max_windows)
    try:
        if memo is None:
            memo = {}
            ms_data._is_targeted_memo = memo
        memo[key] = result
    except AttributeError:
        pass
    return result


def _is_targeted(spec, ms_level, max_windows):
    sel = spec[spec['ms_level'] == ms_level] if 'ms_level' in spec else spec
    if not len(sel) or 'precursor_mz' not in sel:
        return False
    if int(sel['precursor_mz'].round(3).nunique()) > max_windows:
        return False
    return not _windows_tile(sel)


TILE_FRACTION = 0.8


def _windows_tile(sel):
    """Do the isolation windows of these scans tile the m/z range?"""
    cols = ['isolation_lower_mz', 'isolation_upper_mz']
    if not all(c in sel for c in cols):
        return False
    w = (sel[cols].round(4).drop_duplicates()
         .sort_values(cols).to_numpy(dtype=float))
    if len(w) < 3:
        return False
    width = w[:, 1] - w[:, 0]
    gap = w[1:, 0] - w[:-1, 1]
    narrow = np.minimum(width[1:], width[:-1])
    abut = (gap <= 0.5) & (gap >= -0.5 * narrow)
    return bool(abut.mean() >= TILE_FRACTION)


def target_scans(ms_data, precursor_mz, *, tol=DEFAULT_MATCH_TH, ms_level=2,
                 nce=None, rt_range=None):
    """The scans of ONE target, matched on the set precursor mass."""
    spec = ms_data.spectrum_df
    if 'precursor_mz' not in spec.columns:
        raise ValueError('spectrum_df has no precursor_mz column; this is '
                         'not a targeted acquisition')
    mask = np.abs(spec['precursor_mz'].to_numpy() - float(precursor_mz)) <= tol
    if ms_level is not None and 'ms_level' in spec.columns:
        mask &= spec['ms_level'].to_numpy() == ms_level
    if nce is not None and 'nce' in spec.columns:
        values = [nce] if np.isscalar(nce) else list(nce)
        col = spec['nce'].to_numpy()
        hit = np.zeros(len(spec), dtype=bool)
        for v in values:
            hit |= np.abs(col - float(v)) < 0.5
        mask &= hit
    if rt_range is not None:
        rt = spec['rt'].to_numpy()
        mask &= (rt >= rt_range[0]) & (rt <= rt_range[1])
    if not mask.any():
        near = spec.loc[spec['ms_level'] == ms_level, 'precursor_mz'] \
            if 'ms_level' in spec.columns else spec['precursor_mz']
        got = sorted(near.round(3).unique())[:12] if len(near) else []
        raise ValueError(
            f'no scan within {tol} Th of precursor_mz={precursor_mz}; '
            f'the file carries {got}'
            + (' ...' if len(got) == 12 else ''))
    return _SpectrumView(spec[mask], ms_data.peak_df)


def assert_targets_resolved(ms_data, precursor_mz, *, tol=DEFAULT_MATCH_TH,
                            ms_level=2, names=None):
    """Check the match tolerance isolates every target, and raise if not."""
    spec = ms_data.spectrum_df
    if 'precursor_mz' not in spec.columns:
        raise ValueError('spectrum_df has no precursor_mz column; this is '
                         'not a targeted acquisition')
    if ms_level is not None and 'ms_level' in spec.columns:
        spec = spec[spec['ms_level'] == ms_level]
    recorded = spec['precursor_mz'].to_numpy()
    wanted = np.atleast_1d(np.asarray(precursor_mz, dtype=float))
    labels = list(names) if names is not None else [f'{m:.5f}' for m in wanted]

    bad = {}
    for label, m in zip(labels, wanted):
        hit = np.unique(recorded[np.abs(recorded - m) <= tol])
        if hit.size != 1:
            bad[label] = [round(float(h), 5) for h in hit]
    if bad:
        raise ValueError(
            f'tol={tol} Th does not isolate these targets (each must match '
            f'exactly one recorded precursor mass): {bad}. Tighten it below '
            'half the closest spacing in the method.')


def targeted_xic(ms_data, precursor_mz, product_mz, *, tol=DEFAULT_MATCH_TH,
                 mz_tol='auto', mz_tol_unit='ppm', ms_level=2, nce=None,
                 rt_range=None, recentre=True):
    """Chromatogram of one or more transitions of a single target."""
    mz_tol = resolve_mz_tol(mz_tol, mz_tol_unit, ms_data, ms_level=ms_level)
    view = target_scans(ms_data, precursor_mz, tol=tol, ms_level=ms_level,
                        nce=nce, rt_range=rt_range)
    return extract_xic(view, product_mz, mz_tol=mz_tol,
                       mz_tol_unit=mz_tol_unit, ms_level=ms_level,
                       recentre=recentre)
