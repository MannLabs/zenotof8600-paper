"""Total ion current: per spectrum, and as a chromatogram per MS level."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

PER_CHOICES = ('cycle', 'spectrum')


def tic_per_spectrum(ms_data):
    """Summed peak intensity of every spectrum, in spectrum_df row order."""
    inten = ms_data.peak_df['intensity'].to_numpy()
    start = ms_data.spectrum_df['peak_start_idx'].to_numpy(np.int64)
    stop = ms_data.spectrum_df['peak_stop_idx'].to_numpy(np.int64)
    out = np.zeros(len(start), dtype=float)
    ne = np.nonzero(stop > start)[0]
    if not ne.size:
        return out
    s, e = start[ne], stop[ne]
    if np.all(s[1:] == e[:-1]):
        out[ne] = np.add.reduceat(inten[s[0]:e[-1]], s - s[0],
                                  dtype=np.float64)
    else:
        csum = np.concatenate(([0.0], np.cumsum(inten, dtype=np.float64)))
        out[ne] = csum[e] - csum[s]
    return out


def tic(ms_data, *, ms_level=None, per='cycle', rt_range=None, warn=True):
    """TIC chromatogram."""
    if per not in PER_CHOICES:
        raise ValueError(f'per must be one of {PER_CHOICES}, got {per!r}')
    sd = ms_data.spectrum_df
    rt = sd['rt'].to_numpy(float)
    level = (sd['ms_level'].to_numpy() if 'ms_level' in sd.columns
             else np.full(len(sd), -1))
    t = tic_per_spectrum(ms_data)
    order = np.argsort(rt, kind='stable')
    rt, level, t = rt[order], level[order], t[order]
    sel = np.ones(len(rt), bool) if ms_level is None else level == ms_level

    if per == 'cycle' and not (level == 1).any():
        if warn:
            warnings.warn('no MS1 scans to delimit cycles (a targeted method '
                          'without a survey scan?) - returning one point per '
                          'spectrum', stacklevel=2)
        per = 'spectrum'

    if per == 'spectrum':
        out = pd.DataFrame({'rt': rt[sel], 'tic': t[sel], 'n_spectra': 1,
                            'tic_per_spectrum': t[sel],
                            'ms_level': level[sel]})
        ms2_range = (np.nan, np.nan)
    else:
        cycle = np.cumsum(level == 1)
        keep = sel & (cycle > 0)
        n_cyc = int(cycle.max())
        tot = np.bincount(cycle[keep], weights=t[keep],
                          minlength=n_cyc + 1)[1:]
        cnt = np.bincount(cycle[keep], minlength=n_cyc + 1)[1:]
        n2 = np.bincount(cycle[(level == 2) & (cycle > 0)],
                         minlength=n_cyc + 1)[1:]
        full = n2[:-1] if n2.size > 1 else n2
        ms2_range = ((int(full.min()), int(full.max())) if full.size
                     else (0, 0))
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        out = pd.DataFrame({'rt': rt[level == 1], 'tic': tot,
                            'n_spectra': cnt, 'tic_per_spectrum': mean})
        if (warn and ms_level in (None, 2) and full.size
                and ms2_range[0] != ms2_range[1]):
            warnings.warn(
                f'MS2 scans per cycle vary from {ms2_range[0]} to '
                f'{ms2_range[1]} (DDA, or a scheduled targeted method): the '
                f"per-cycle sum follows how many scans ran - see "
                f"'tic_per_spectrum', or per='spectrum'", stacklevel=2)

    if rt_range is not None:
        out = out[(out['rt'] >= rt_range[0])
                  & (out['rt'] <= rt_range[1])].reset_index(drop=True)
    out.attrs['per'] = per
    out.attrs['ms_level'] = ms_level
    out.attrs['ms2_per_cycle'] = ms2_range
    return out
