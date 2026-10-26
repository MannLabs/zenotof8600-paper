"""Per-run instrument metrics: one row per run, one number per question."""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from .ion_current import tic_per_spectrum

METRICS = [
    ('n_precursors', 'depth', 'Precursors', '#', 'search', +1,
     'distinct precursor ids with a non-null intensity'),
    ('n_peptides', 'depth', 'Peptides', '#', 'search', +1,
     'distinct peptidoforms (sequence + modifications)'),
    ('n_protein_groups', 'depth', 'Protein groups', '#', 'search', +1,
     'distinct protein groups carrying a quantity in this run'),
    ('signal_ms2', 'signal', 'MS2 area', '% vs run mean', 'raw', +1,
     'median over targets of each target summed-fragment peak AREA, as a '
     'log2 ratio to that target mean across runs; unnormalised. Area, not '
     'apex: an apex is a height and carries the peak width'),
    ('signal_ms1', 'signal', 'MS1 area', '% vs run mean', 'raw', +1,
     'as signal_ms2 but from the precursor XIC; weaker and more interfered, '
     'kept as a contrast'),
    ('ms2_tic_median', 'signal', 'MS2 TIC', 'counts/cycle', 'raw', +1,
     'median per-cycle summed MS2 ion current — target-free signal level'),
    ('ms1_tic_median', 'signal', 'MS1 TIC', 'counts/cycle', 'raw', +1,
     'median MS1 total ion current per cycle'),
    ('normalised_intensity', 'signal', 'Export intensity', 'a.u.', 'search', 0,
     'median engine intensity over precursors seen in every run — COMPOSITION, '
     'not sensitivity, whenever normalised_scale_spread is small'),
    ('normalised_scale_spread', 'signal', 'Export scale spread', 'log2',
     'search', 0,
     'spread of per-run median log2 ratios; < 0.10 while counts vary means the '
     'export is cross-run normalised and its intensities cannot measure signal'),
    ('stability_ms2', 'stability', 'MS2 instability', '%', 'raw', -1,
     'high-frequency scatter of the per-cycle summed MS2 ion current, from '
     'successive differences; spray flicker is common mode across the windows '
     'of a cycle while chromatography is not'),
    ('stability_ms1', 'stability', 'MS1 instability', '%', 'raw', -1,
     'the same on the MS1 TIC; one scan per cycle, so ~3x noisier'),
    ('frac_z2', 'chemistry', 'Charge 2+', 'fraction', 'search', 0,
     'fraction of precursors at charge 2'),
    ('frac_z3', 'chemistry', 'Charge 3+', 'fraction', 'search', 0, ''),
    ('frac_z4plus', 'chemistry', 'Charge >=4', 'fraction', 'search', 0, ''),
    ('frac_mc0', 'chemistry', 'Fully cleaved', 'fraction', 'search', +1,
     'fraction of peptides with no missed cleavage'),
    ('median_peptide_length', 'chemistry', 'Peptide length', 'aa', 'search', 0,
     ''),
    ('acq_time', 'timing', 'Acquisition time', 'epoch s', 'raw', 0,
     'raw-file mtime; a proxy, and a file copy resets it'),
    ('acq_order', 'timing', 'Acquisition order', 'rank', 'raw', 0,
     'rank of acq_time — derive order from the files, never from the queue'),
]

METRIC_INDEX = {m[0]: m for m in METRICS}


NORM_FLAT_LOG2 = 0.10
NORM_COUNT_SPREAD = 0.10


MIN_SCANS_IN_PROBE = 12


def seed_rt_window(cycle_s, *, start=0.05, min_scans=MIN_SCANS_IN_PROBE,
                   verbose=True, label='rt window'):
    """Raise a window search's first probe so it can hold `min_scans` scans."""
    start = float(start)
    if not np.isfinite(cycle_s) or cycle_s <= 0:
        return start
    need = min_scans * cycle_s / 60.0 / 2.0
    if need <= start:
        return start
    if verbose:
        held = 2 * start * 60.0 / cycle_s
        print(f'  {label} first probe {start:.3f} -> {need:.3f} min: at a '
              f'{cycle_s:.2f} s cycle the default holds only {held:.1f} '
              f'scans, under the {min_scans} a peak and its baseline need')
    return float(need)


def auto_rt_window(measure, *, start=0.05, factor=3.0, bounds=(0.02, 0.50),
                   rel_tol=0.10, max_iter=5, max_expand=6, expand=2.0,
                   min_ok_frac=0.25, verbose=True, label='rt window'):
    """Choose an extraction half-window from the peaks it will be used to measure."""
    lo, hi = bounds
    rt_tol, rows = float(np.clip(start, lo, hi)), []
    expansions, widened, converged, anchored = 0, [], False, False
    thin_rows = []
    for _ in range(max_iter + max_expand):
        got = measure(rt_tol)
        fwhm, info = got if isinstance(got, tuple) else (got, {})
        measurable = not (fwhm is None or not np.isfinite(fwhm) or fwhm <= 0)
        is_thin = (measurable and np.isfinite(info.get('ok_frac', np.nan))
                   and info['ok_frac'] < min_ok_frac)
        if not measurable or is_thin:
            if rows:
                anchored = True
                break
            if is_thin:
                thin_rows.append({'rt_tol': rt_tol, 'fwhm_s': fwhm * 60,
                                  **info})
            if rt_tol >= hi or expansions >= max_expand:
                if thin_rows:
                    rows.append(thin_rows[-1])
                    rt_tol = float(np.clip(
                        factor * thin_rows[-1]['fwhm_s'] / 60, lo, hi))
                    break
                if verbose:
                    tried = ', '.join(f'{w:.3f}' for w in widened + [rt_tol])
                    print(f'  {label} auto: no measurable peaks at any window '
                          f'in {lo:.3f}-{hi:.3f} min (tried {tried}) — '
                          f'returning NaN rather than an untested window')
                return float('nan'), pd.DataFrame(rows)
            widened.append(rt_tol)
            expansions += 1
            rt_tol = float(min(rt_tol * expand, hi))
            continue
        rows.append({'rt_tol': rt_tol, 'fwhm_s': fwhm * 60, **info})
        nxt = float(np.clip(factor * fwhm, lo, hi))
        converged = abs(nxt - rt_tol) <= rel_tol * rt_tol
        rt_tol = nxt
        if converged:
            break
    trace = pd.DataFrame(rows)
    trace.attrs['converged'] = bool(converged or anchored)
    trace.attrs['rt_tol'] = float(rt_tol)
    if verbose and len(trace):
        steps = ' -> '.join(f'{r.rt_tol:.3f} (FWHM {r.fwhm_s:.2f} s)'
                            for r in trace.itertuples())
        skipped = (f'unmeasurable at {", ".join(f"{w:.3f}" for w in widened)}; '
                   if widened else '')
        print(f'  {label} auto: {skipped}{steps}  =>  {rt_tol:.3f} min')
    if not (converged or anchored) and len(trace):
        last = (abs(rt_tol - trace['rt_tol'].iloc[-1])
                / trace['rt_tol'].iloc[-1] * 100)
        warnings.warn(
            f'{label}: window search did not converge in {max_iter} '
            f'iterations — the last step still moved {last:.0f}% (tolerance '
            f'{rel_tol * 100:.0f}%). Returning {rt_tol:.3f} min, which the '
            f'search was still expanding, so any width measured with it is '
            f'biased NARROW. Raise max_iter, or pin the window from a sweep.',
            stacklevel=2)
    return rt_tol, trace


