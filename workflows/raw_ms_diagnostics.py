"""MS1 precursor and MS2 fragment diagnostics measured from the raw files."""

from __future__ import annotations

import gc
import os
import re

import warnings

import numpy as np
import pandas as pd
from zenotof8600_analytics import sigma_iqr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import technical_qc_analysis as tqc
from zenotof8600_analytics.raw import merge_peaks, stratified_sample


USE_RAW = False

FORCE = False

N_PRECURSORS = 2000

TOP_INTENSITY_FRAC = 0.30
MZ_BINS = 10
ABUNDANCE_BINS = 10

RUNS_PER_CONDITION = 1

RT_WINDOW = 0.05
MS2_SCAN_SEARCH = 8

PEAK_PICK_FRAGMENTS = 8

MIN_COELUTING_FRAGMENTS = 3
COELUTION_TOL = 0.02

SUM_MS2_SCANS = True

SUM_MERGE_PPM = 1.0

CACHE_DIR = None
CACHE_SUBDIR = 'ms_data_cache_profile'

SEARCH_PPM = 30.0
SEARCH_ABS_TH = 0.02

MIN_SN = 3.0

MIN_SIDE_SAMPLES = 2

CENTROID_NOISE_PPM = 500.0
CENTROID_NOISE_TH = 0.5
CENTROID_METHOD = 'fwhm_mid'
CENTROID_FRAC = 0.1

CALIBRATE = True
CALIB_TOL_PPM = 10.0
CALIB_SEARCH_PPM = 40.0
CALIB_INTENSITY_FRAC = 0.5
CALIB_MAX_TARGETS = 1000
CALIB_MZ_BINS = 20
CALIB_ABUNDANCE_BINS = 5
CALIB_DELTA_FRAC = 0.01
CALIB_MIN_POINTS = 150
CALIB_FRAC = 0.4

MAX_ISOTOPES = 8
MIN_ISO_ABUNDANCE = 0.01
ISOTOPE_SPACING = 1.00335483
MEASURE_MS1_ISOTOPES = True

FRAG_TYPES = ('b', 'y')
MAX_FRAGMENTS_PER_PRECURSOR = 12
MIN_FRAG_MZ = 200.0

MOD_BY_DELTA = {
    '+57.02': 'Carbamidomethyl@C',
    '+15.99': 'Oxidation@M',
    '+42.01': 'Acetyl@Protein_N-term',
}

_RNG_SEED = 0
_PANEL_LEVELS = ('MS1 precursor', 'MS2 fragment')


OUT_SUBDIR = 'raw_ms_diagnostics'
SPECTRA_SUBDIR = 'spectra'


def _out_dir():
    d = os.path.join(tqc.DATA_ROOT, OUT_SUBDIR)
    os.makedirs(d, exist_ok=True)
    return d


def save(fig, name, source_df=None, subdir=None):
    """Same output contract as the QC set (PDF + 300 dpi PNG + source data), written to this script's own subfolder."""
    out = _out_dir() if subdir is None else os.path.join(_out_dir(), subdir)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, f'{name}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(out, f'{name}.png'), dpi=300, bbox_inches='tight')
    if tqc.SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)
    if source_df is not None:
        source_df.to_csv(os.path.join(out, f'{name}_sourcedata.csv'), index=False)
    print(f'  wrote {name}.pdf  + sourcedata -> {out}')


def _profile_cache_dir():
    """Where the full-profile HDF caches go: ``<data root>/<CACHE_SUBDIR>``."""
    if CACHE_DIR is not None:
        return CACHE_DIR
    root = getattr(tqc, 'DATA_ROOT', None)
    if root:
        return os.path.join(root, CACHE_SUBDIR)
    base = (os.environ.get('LOCALAPPDATA')
            or os.path.join(os.path.expanduser('~'), '.cache'))
    return os.path.join(base, 'zenotof8600_analytics', CACHE_SUBDIR)


def configure(*, use_raw=None, force=None, n_precursors=None,
              top_intensity_frac=None,
              runs_per_condition=None, cache_dir=None, extra_conditions=None,
              **tqc_kwargs):
    """Set config; unknown keywords go to tqc.configure."""
    global USE_RAW, N_PRECURSORS, RUNS_PER_CONDITION, CACHE_DIR, FORCE, TOP_INTENSITY_FRAC
    if use_raw is not None:
        USE_RAW = bool(use_raw)
    if force is not None:
        FORCE = bool(force)
    if n_precursors is not None:
        N_PRECURSORS = int(n_precursors)
    if top_intensity_frac is not None:
        TOP_INTENSITY_FRAC = float(top_intensity_frac)
    if runs_per_condition is not None:
        RUNS_PER_CONDITION = int(runs_per_condition)
    if cache_dir is not None:
        CACHE_DIR = cache_dir
    result = tqc.configure(**tqc_kwargs)
    if extra_conditions:
        import mass_accuracy_analysis as ma
        ma._register_extra_conditions(extra_conditions)
    return result


def parse_peaks_peptide(peptide):
    """'AAAAC(+57.02)LDK' -> ('AAAACLDK', [(5, 'Carbamidomethyl@C')])."""
    seq, mods, pos = [], [], 0
    for token in re.findall(r'[A-Z]|\([+-][\d.]+\)', str(peptide)):
        if token.startswith('('):
            name = MOD_BY_DELTA.get(token[1:-1])
            if name is None:
                return None, None
            mods.append((max(pos, 1), name))
        else:
            seq.append(token)
            pos += 1
    return ''.join(seq), mods


def _residue_formula(seq, mods):
    """Elemental composition of a residue stretch plus its modifications."""
    from alphabase.peptide.precursor import get_mod_seq_formula
    return get_mod_seq_formula(seq, ';'.join(name for _, name in mods))


_ISO_CALC = None


def theoretical_isotopes(formula, n_max=None):
    """Relative isotope abundances from the monoisotopic peak upwards."""
    global _ISO_CALC
    if _ISO_CALC is None:
        from alphabase.constants.isotope import IsotopeDistribution
        _ISO_CALC = IsotopeDistribution()
    dist, mono_idx = _ISO_CALC.calc_formula_distribution(formula)
    iso = np.asarray(dist[mono_idx:], dtype=float)
    if iso.size == 0 or iso[0] <= 0:
        return None
    iso = iso / iso.max()
    keep = np.where(iso >= MIN_ISO_ABUNDANCE)[0]
    n = min(int(keep.max()) + 1 if keep.size else 1,
            MAX_ISOTOPES if n_max is None else n_max)
    return iso[:max(n, 2)]


def _formula_mass(formula):
    """Monoisotopic mass of a [(element, count), ...] list."""
    from alphabase.constants.atom import CHEM_MONO_MASS
    return float(sum(CHEM_MONO_MASS[elem] * n for elem, n in formula))


def fragment_targets(seq, mods, precursor_mz, charge, *, above_precursor=True):
    """b/y fragments as (label, m/z, charge, formula)."""
    PROTON = 1.00727646677
    out = []
    n = len(seq)
    for ion in FRAG_TYPES:
        for i in range(2, n):
            if ion == 'b':
                part, part_mods = seq[:i], [(p, m) for p, m in mods if p <= i]
                extra = []
            else:
                part = seq[n - i:]
                part_mods = [(p, m) for p, m in mods if p > n - i]
                extra = [('H', 2), ('O', 1)]
            formula = _residue_formula(part, part_mods) + extra
            mz = _formula_mass(formula) + PROTON
            if mz < MIN_FRAG_MZ or (above_precursor and mz <= precursor_mz):
                continue
            out.append((f'{ion}{i}', mz, 1, formula))
    return out


def _fit_peak_centre(mz, inten, *, method='guo'):
    """Sub-sample peak centre from a MODEL of the peak, not from its samples."""
    y = np.asarray(inten, dtype=float)
    if y.size < 4 or np.any(y <= 0):
        ok = y > 0
        mz, y = np.asarray(mz, float)[ok], y[ok]
        if y.size < 4:
            return np.nan
    mz = np.asarray(mz, dtype=float)
    x0 = float(mz.mean())
    x = mz - x0
    w = y * y
    L = np.log(y)
    S = np.array([
        [(w * x ** 4).sum(), (w * x ** 3).sum(), (w * x ** 2).sum()],
        [(w * x ** 3).sum(), (w * x ** 2).sum(), (w * x).sum()],
        [(w * x ** 2).sum(), (w * x).sum(), w.sum()],
    ])
    v = np.array([(w * L * x ** 2).sum(), (w * L * x).sum(), (w * L).sum()])
    try:
        a, b, c = np.linalg.solve(S, v)
    except np.linalg.LinAlgError:
        return np.nan
    if not (a < 0):
        return np.nan
    centre = -b / (2.0 * a)
    if abs(centre) > (x.max() - x.min()):
        return np.nan

    if method == 'gaussfit':
        try:
            from scipy.optimize import curve_fit

            sigma0 = float(np.sqrt(-1.0 / (2.0 * a)))
            amp0 = float(np.exp(c - b * b / (4.0 * a)))

            def _g(t, amp, mu, sig):
                return amp * np.exp(-0.5 * ((t - mu) / sig) ** 2)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                popt, _ = curve_fit(_g, x, y, p0=[amp0, centre, sigma0],
                                    maxfev=2000)
            if abs(popt[1]) <= (x.max() - x.min()) and popt[2] > 0:
                centre = float(popt[1])
        except Exception:
            pass
    return centre + x0


def measure_profile_peak(mz, inten, target, *, tol_ppm=None, offset_ppm=0.0):
    """Measure one profile peak near `target`."""
    centre = target * (1.0 + offset_ppm * 1e-6)
    tight = centre * (SEARCH_PPM if tol_ppm is None else tol_ppm) * 1e-6
    wide = max(centre * SEARCH_PPM * 1e-6 * 4, SEARCH_ABS_TH)
    lo, hi = np.searchsorted(mz, [centre - wide, centre + wide])
    if hi - lo < 5:
        return None
    m = mz[lo:hi]
    y = inten[lo:hi].astype(float)

    inner = np.where(np.abs(m - centre) <= tight)[0]
    if inner.size == 0:
        return None
    k = int(inner[np.argmax(y[inner])])
    if k == 0 or k == len(y) - 1:
        return None
    base = float(np.percentile(y, 5))
    apex = float(y[k])
    apex_fit = apex
    if (0 < k < len(y) - 1) and y[k - 1] > base and y[k + 1] > base:
        d2 = y[k - 1] - 2.0 * y[k] + y[k + 1]
        if d2 < 0:
            d1 = y[k + 1] - y[k - 1]
            cand = float(y[k] - d1 * d1 / (8.0 * d2))
            if apex < cand <= 1.25 * apex:
                apex_fit = cand
    noise = float(np.median(y))
    if apex <= 0 or (apex - base) < MIN_SN * max(noise, 1e-9):
        return None

    hm = base + 0.5 * ((apex_fit if CENTROID_METHOD == 'fwhm_mid' else apex)
                       - base)
    i = k
    while i > 0 and y[i] > hm:
        i -= 1
    j = k
    while j < len(y) - 1 and y[j] > hm:
        j += 1
    if y[i] > hm or y[j] > hm:
        return None

    left = np.interp(hm, [y[i], y[i + 1]], [m[i], m[i + 1]])
    right = np.interp(hm, [y[j], y[j - 1]], [m[j], m[j - 1]])
    fwhm = float(right - left)
    if not np.isfinite(fwhm) or fwhm <= 0:
        return None

    centroid = np.nan
    if CENTROID_METHOD == 'fwhm_mid':
        if (k - i) < MIN_SIDE_SAMPLES or (j - k) < MIN_SIDE_SAMPLES:
            return None
        centroid = 0.5 * (left + right)
    if CENTROID_METHOD == 'gauss' and 0 < k < len(y) - 1:
        y0, y1, y2 = y[k - 1] - base, y[k] - base, y[k + 1] - base
        if min(y0, y1, y2) > 0:
            l0, l1, l2 = np.log(y0), np.log(y1), np.log(y2)
            denom = l0 - 2 * l1 + l2
            if denom < 0:
                shift = 0.5 * (l0 - l2) / denom
                if abs(shift) <= 1:
                    centroid = float(m[k] + shift * 0.5 * (m[k + 1] - m[k - 1]))
    if CENTROID_METHOD in ('guo', 'gaussfit') and not np.isfinite(centroid):
        lim = base + CENTROID_FRAC * (apex - base)
        a2, b2 = k, k
        while a2 > 0 and y[a2 - 1] > lim:
            a2 -= 1
        while b2 < len(y) - 1 and y[b2 + 1] > lim:
            b2 += 1
        xs, ys = m[a2:b2 + 1], y[a2:b2 + 1] - base
        good = ys > 0
        if int(good.sum()) >= 4:
            centroid = _fit_peak_centre(xs[good], ys[good],
                                        method=CENTROID_METHOD)
    if not np.isfinite(centroid):
        lim = (base + CENTROID_FRAC * (apex - base)
               if CENTROID_METHOD in ('frac', 'guo', 'gaussfit') else hm)
        a2, b2 = k, k
        while a2 > 0 and y[a2 - 1] > lim:
            a2 -= 1
        while b2 < len(y) - 1 and y[b2 + 1] > lim:
            b2 += 1
        top = slice(a2, b2 + 1)
        w = y[top] - base
        if w.sum() <= 0:
            return None
        centroid = float((m[top] * w).sum() / w.sum())
    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    area = float(_trapz(np.clip(y - base, 0, None), m))
    return {'mz': centroid, 'fwhm': fwhm,
            'height': (apex_fit if CENTROID_METHOD == 'fwhm_mid' else apex) - base,
            'area': area,
            'sn': (apex - base) / max(noise, 1e-9)}


def measure_centroid_peak(mz, inten, target, *, tol_ppm=None, offset_ppm=0.0):
    """Nearest centroid to `target`, for spectra that carry no peak shape."""
    centre = target * (1.0 + offset_ppm * 1e-6)
    half = centre * (SEARCH_PPM if tol_ppm is None else tol_ppm) * 1e-6
    lo, hi = np.searchsorted(mz, [centre - half, centre + half])
    if hi <= lo:
        return None
    m, y = mz[lo:hi], inten[lo:hi].astype(float)
    k = int(np.argmax(y))
    wide = max(centre * CENTROID_NOISE_PPM * 1e-6, CENTROID_NOISE_TH)
    a, b = np.searchsorted(mz, [centre - wide, centre + wide])
    nb = np.concatenate([inten[a:lo], inten[hi:b]]).astype(float)
    noise = float(np.median(nb)) if nb.size >= 5 else float(np.median(y))
    if y[k] <= 0 or y[k] < MIN_SN * max(noise, 1e-9):
        return None
    return {'mz': float(m[k]), 'fwhm': np.nan, 'height': float(y[k]),
            'area': float(y[k]), 'sn': float(y[k] / max(noise, 1e-9))}


def measure_peak(mz, inten, target, *, profile=True, tol_ppm=None,
                 offset_ppm=0.0):
    """Measure a peak, per peak, without deciding up front what the data is."""
    if profile:
        r = measure_profile_peak(mz, inten, target, tol_ppm=tol_ppm,
                                 offset_ppm=offset_ppm)
        if r is not None:
            return r
    return measure_centroid_peak(mz, inten, target, tol_ppm=tol_ppm,
                                 offset_ppm=offset_ppm)


def is_profile_spectrum(mz, inten, *, min_points=4):
    """Heuristic: profile spectra put several samples across one peak."""
    if len(mz) < 10:
        return False
    apex = mz[int(np.argmax(inten))]
    return int((np.abs(mz - apex) < 0.02).sum()) >= min_points


def measure_isotopes(mz, inten, mono_mz, charge, n_iso, *, profile=True,
                     tol_ppm=None, calib=None):
    """Areas (profile) or heights (centroid) of the isotope series."""
    areas = np.zeros(n_iso)
    for k in range(n_iso):
        iso_mz = mono_mz + k * ISOTOPE_SPACING / charge
        peak = measure_peak(mz, inten, iso_mz, profile=profile,
                            tol_ppm=tol_ppm,
                            offset_ppm=0.0 if calib is None else calib(iso_mz))
        if peak is not None:
            areas[k] = peak['area']
    return areas


def cosine_similarity(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


class MassCalibration:
    """Mass deviation in ppm as a smooth function of m/z, fit by LOESS."""

    def __init__(self, mz, ppm, *, frac=None, label=''):
        from statsmodels.nonparametric.smoothers_lowess import lowess

        mz = np.asarray(mz, float)
        ppm = np.asarray(ppm, float)
        ok = np.isfinite(mz) & np.isfinite(ppm)
        mz, ppm = mz[ok], ppm[ok]
        if mz.size:
            med = np.median(ppm)
            sig = (np.quantile(ppm, .75) - np.quantile(ppm, .25)) / 1.349
            keep = np.abs(ppm - med) <= max(5 * sig, 1.0)
            mz, ppm = mz[keep], ppm[keep]
        self.n = int(mz.size)
        self.label = label
        self.median = float(np.median(ppm)) if self.n else 0.0
        if self.n < CALIB_MIN_POINTS:
            self._x = self._y = None
            return
        span = float(np.ptp(mz)) if mz.size else 0.0
        fit = lowess(ppm, mz, frac=frac or CALIB_FRAC, it=3,
                     return_sorted=True, delta=CALIB_DELTA_FRAC * span)
        x, y = fit[:, 0], fit[:, 1]
        uniq, first = np.unique(x, return_index=True)
        self._x, self._y = uniq, y[first]

    def __call__(self, mz):
        if self._x is None:
            return self.median
        return float(np.interp(float(mz), self._x, self._y,
                               left=self._y[0], right=self._y[-1]))

    def describe(self):
        if self._x is None:
            return f'{self.label}: n={self.n} (<{CALIB_MIN_POINTS}), constant {self.median:+.2f} ppm'
        return (f'{self.label}: n={self.n:,}  offset {self.median:+.2f} ppm, '
                f'curve {self._y.min():+.2f} to {self._y.max():+.2f} ppm '
                f'over m/z {self._x[0]:.0f}-{self._x[-1]:.0f}')


def _fit_calibrations(rows, verbose=True):
    """Fit one calibration per level from a wide-window first pass."""
    out = {}
    if not len(rows):
        return out
    d = pd.DataFrame(rows)
    for level in _PANEL_LEVELS:
        g = d[d['level'] == level]
        if 'label' in g:
            mono = g[g['label'] == 'precursor']
            if len(mono) >= CALIB_MIN_POINTS:
                g = mono
        if not len(g):
            continue
        cal = MassCalibration(g['mz_theoretical'], g['ppm'], label=level)
        out[level] = cal
        if verbose:
            print(f'    calib {cal.describe()}')
    return out


def sample_precursors(df, n=None, *, seed=_RNG_SEED):
    """Stratified sample over an m/z x intensity grid."""
    n = N_PRECURSORS if n is None else n
    d = df.dropna(subset=['mz_measured', 'rt', 'intensity', 'sequence'])
    return stratified_sample(
        d, n, mz_col='mz_measured', intensity_col='intensity',
        mz_bins=MZ_BINS, abundance_bins=ABUNDANCE_BINS,
        top_fraction=TOP_INTENSITY_FRAC, seed=seed)

def calibration_subset(targets, n=None, *, top_fraction=None):
    """The targets the calibration curve is fitted on."""
    n = CALIB_MAX_TARGETS if n is None else n
    frac = CALIB_INTENSITY_FRAC if top_fraction is None else top_fraction
    sub = stratified_sample(
        targets, n, mz_col='mz_theoretical', intensity_col='intensity',
        mz_bins=CALIB_MZ_BINS, abundance_bins=CALIB_ABUNDANCE_BINS,
        top_fraction=frac, seed=_RNG_SEED)
    return sub if len(sub) else targets.head(n)


def precursor_table(features_path, run):
    """Per-run precursor table from a PEAKS features.csv."""
    df = pd.read_csv(features_path, low_memory=False)
    out = pd.DataFrame({
        'peptide': df['Peptide'].astype(str),
        'charge': pd.to_numeric(df['z'], errors='coerce'),
        'mz_theoretical': pd.to_numeric(df['m/z'], errors='coerce'),
        'mz_measured': pd.to_numeric(df.get(f'{run} m/z'), errors='coerce'),
        'rt': pd.to_numeric(df.get(f'{run} RT mean'), errors='coerce'),
        'intensity': pd.to_numeric(df.get(f'{run} Normalized Area'), errors='coerce'),
    })
    seqs, modlists = zip(*(parse_peaks_peptide(p) for p in out['peptide']))
    out['sequence'] = seqs
    out['mods'] = list(modlists)
    return out.dropna(subset=['mz_measured', 'rt', 'charge'])


class _ScanIndex:
    """Everything the measurement loop needs, as numpy arrays, built once."""

    def __init__(self, ms):
        sdf = ms.spectrum_df
        self.rt = sdf['rt'].to_numpy(float)
        self.level = sdf['ms_level'].to_numpy()
        self.start = sdf['peak_start_idx'].to_numpy(np.int64)
        self.stop = sdf['peak_stop_idx'].to_numpy(np.int64)
        self.mz = ms.peak_df['mz'].to_numpy()
        self.inten = ms.peak_df['intensity'].to_numpy()

        pos1 = np.nonzero(self.level == 1)[0]
        self.ms1_pos = pos1[np.argsort(self.rt[pos1])]
        self.ms1_rt = self.rt[self.ms1_pos]

        pos2 = np.nonzero(self.level == 2)[0]
        lo = sdf['isolation_lower_mz'].to_numpy(float)[pos2]
        hi = sdf['isolation_upper_mz'].to_numpy(float)[pos2]
        order = np.lexsort((self.rt[pos2], hi, lo))
        lo, hi, pos2 = lo[order], hi[order], pos2[order]
        bounds = (np.nonzero((np.diff(lo) != 0) | (np.diff(hi) != 0))[0] + 1
                  if len(lo) else np.array([], dtype=int))
        starts = np.concatenate([[0], bounds, [len(lo)]]) if len(lo) else [0]
        self.win_lo, self.win_hi, self.win_scans = [], [], []
        for a, b in zip(starts[:-1], starts[1:]):
            self.win_lo.append(lo[a])
            self.win_hi.append(hi[a])
            self.win_scans.append((self.rt[pos2[a:b]], pos2[a:b]))
        self.win_lo = np.asarray(self.win_lo, float)
        self.win_hi = np.asarray(self.win_hi, float)

    def spectrum(self, p):
        s, e = self.start[p], self.stop[p]
        return self.mz[s:e], self.inten[s:e]

    def ms2_scans(self, mz, rt, rt_window, limit=None):
        """Scans of the precursor's isolation window(s) within rt_window of rt, nearest first."""
        rts, pos = [], []
        for w in np.nonzero((self.win_lo <= mz) & (self.win_hi >= mz))[0]:
            wrt, wpos = self.win_scans[w]
            a, b = np.searchsorted(wrt, [rt - rt_window, rt + rt_window])
            if b > a:
                rts.append(wrt[a:b])
                pos.append(wpos[a:b])
        if not rts:
            return np.array([]), np.array([], dtype=np.int64)
        rts = np.concatenate(rts)
        pos = np.concatenate(pos)
        order = np.argsort(np.abs(rts - rt))
        if limit is not None:
            order = order[:limit]
        return rts[order], pos[order]


def _fragment_xics(idx, scan_pos, mzs, tol_ppm, *, offsets=None):
    """One XIC per target m/z, in a single pass over the scans."""
    mzs = np.asarray(mzs, float)
    if offsets is not None:
        mzs = mzs * (1.0 + np.asarray(offsets, float) * 1e-6)
    half = mzs * tol_ppm * 1e-6
    lo_t, hi_t = mzs - half, mzs + half
    out = np.zeros((len(mzs), len(scan_pos)), dtype=float)
    from zenotof8600_analytics.raw.extraction import window_indices
    for j, p in enumerate(scan_pos):
        s, e = idx.start[p], idx.stop[p]
        smz, sin = idx.mz[s:e], idx.inten[s:e]
        li, ri = window_indices(smz, lo_t, hi_t)
        for i in range(len(mzs)):
            if ri[i] > li[i]:
                out[i, j] = sin[li[i]:ri[i]].sum()
    return out


def _ms2_scan_index(sdf):
    """Group MS2 scans by isolation window, keyed by DataFrame label."""
    ms2 = sdf[sdf['ms_level'] == 2]
    windows = {}
    for (lo, hi), g in ms2.groupby(['isolation_lower_mz', 'isolation_upper_mz']):
        order = np.argsort(g['rt'].values)
        windows[(float(lo), float(hi))] = (g['rt'].values[order],
                                           g.index.values[order])
    return windows


def _candidate_ms2_scans(windows, mz, rt, rt_window):
    """Scans of the precursor's isolation window within rt_window of rt."""
    out = []
    for (lo, hi), (rts, idx) in windows.items():
        if lo <= mz <= hi:
            sel = np.abs(rts - rt) <= rt_window
            out.extend(zip(rts[sel], idx[sel]))
    out.sort(key=lambda r: abs(r[0] - rt))
    return out[:MS2_SCAN_SEARCH]


def _sum_spectra(spectra):
    """Sum spectra that share a TOF m/z grid."""
    if not spectra:
        return np.array([]), np.array([])
    mz = np.concatenate([s[0] for s in spectra])
    inten = np.concatenate([s[1] for s in spectra])
    scan = np.repeat(np.arange(len(spectra)), [len(s[0]) for s in spectra])
    return merge_peaks(mz, inten, tol=SUM_MERGE_PPM, tol_unit='ppm',
                       position='median', scan_id=scan)


def extract_run(raw_path, targets, *, condition, run, verbose=True):
    """Measure MS1 precursor and MS2 fragment peaks for one run."""
    from zenotof8600_analytics.raw import load_ms_data
    from zenotof8600_analytics.raw.extraction import pick_peak

    ms = load_ms_data(raw_path, cache_dir=_profile_cache_dir(), centroided=False,
                      verbose=verbose)
    idx = _ScanIndex(ms)

    profile_by_level = {}
    if idx.ms1_pos.size:
        votes = [is_profile_spectrum(
                     *idx.spectrum(idx.ms1_pos[int(idx.ms1_pos.size * f)]))
                 for f in (0.3, 0.5, 0.7)]
        profile_by_level[1] = sum(votes) >= 2
    votes = []
    for _wrt, wpos in idx.win_scans[:6]:
        mid = len(wpos) // 2
        take = wpos[max(0, mid - MS2_SCAN_SEARCH // 2):
                    mid + MS2_SCAN_SEARCH // 2 + 1]
        if not len(take):
            continue
        smz, sin = (_sum_spectra([idx.spectrum(p) for p in take])
                    if SUM_MS2_SCANS else idx.spectrum(take[len(take) // 2]))
        votes.append(is_profile_spectrum(smz, sin))
    if votes:
        profile_by_level[2] = sum(votes) > len(votes) / 2
    if verbose:
        print(f'    profile data: MS1={profile_by_level.get(1)}, '
              f'MS2={profile_by_level.get(2)}')

    def measure(subset, cals, tol_ppm):
        """One measurement pass."""
        cal1 = cals.get(_PANEL_LEVELS[0])
        cal2 = cals.get(_PANEL_LEVELS[1])
        rows = []
        for t in subset.itertuples():
            seq, mods = t.sequence, t.mods
            if not seq:
                continue
            z = int(t.charge)

            frags = fragment_targets(seq, mods, t.mz_theoretical, z)
            if not frags:
                continue

            pick = frags[:PEAK_PICK_FRAGMENTS]
            frag_mz = np.array([f[1] for f in pick], float)
            frag_off = (np.array([cal2(m) for m in frag_mz])
                        if cal2 is not None else None)

            rts, pos = idx.ms2_scans(float(t.mz_measured), float(t.rt),
                                     RT_WINDOW)
            if pos.size < 3:
                continue
            order = np.argsort(rts)
            rts, pos = rts[order], pos[order]
            xics = _fragment_xics(idx, pos, frag_mz, tol_ppm, offsets=frag_off)

            totals = xics.sum(axis=1)
            live = totals > 0
            if int(live.sum()) < MIN_COELUTING_FRAGMENTS:
                continue
            top = np.argsort(-totals)[:6]
            peak_rt = pick_peak(rts, xics[top].sum(axis=0), baseline='p5')
            apex_rt = peak_rt.get('apex_rt', np.nan)
            if not np.isfinite(apex_rt):
                continue

            own_apex = rts[np.argmax(xics, axis=1)]
            n_coelute = int((live &
                             (np.abs(own_apex - apex_rt) <= COELUTION_TOL)).sum())
            if n_coelute < MIN_COELUTING_FRAGMENTS:
                continue

            lo_rt = peak_rt.get('left_rt', np.nan)
            hi_rt = peak_rt.get('right_rt', np.nan)
            if not (np.isfinite(lo_rt) and np.isfinite(hi_rt) and hi_rt > lo_rt):
                lo_rt, hi_rt = apex_rt - RT_WINDOW, apex_rt + RT_WINDOW

            centre_rt = 0.5 * (lo_rt + hi_rt)
            _, cand = idx.ms2_scans(float(t.mz_measured), centre_rt,
                                    max(0.5 * (hi_rt - lo_rt), 1e-6),
                                    limit=MS2_SCAN_SEARCH)
            if not cand.size:
                _, cand = idx.ms2_scans(float(t.mz_measured), apex_rt,
                                        RT_WINDOW, limit=MS2_SCAN_SEARCH)
            if not cand.size:
                continue

            if SUM_MS2_SCANS and cand.size > 1:
                fmz, fin = _sum_spectra([idx.spectrum(p) for p in cand])
            else:
                fmz, fin = idx.spectrum(cand[0])

            measured = []
            for label, theo_mz, fz, formula in frags:
                off = float(cal2(theo_mz)) if cal2 is not None else 0.0
                fpeak = measure_peak(fmz, fin, theo_mz, profile=True,
                                     tol_ppm=tol_ppm, offset_ppm=off)
                if fpeak is not None:
                    measured.append((label, theo_mz, fz, formula, fpeak, off))
            measured.sort(key=lambda r: -r[4]['area'])
            for label, theo_mz, fz, formula, fpeak, off in \
                    measured[:MAX_FRAGMENTS_PER_PRECURSOR]:
                iso_theo = theoretical_isotopes(formula)
                cos, n_iso = np.nan, 0
                if iso_theo is not None:
                    obs = measure_isotopes(fmz, fin, theo_mz, fz, len(iso_theo),
                                           profile=True, tol_ppm=tol_ppm,
                                           calib=cal2)
                    cos = cosine_similarity(obs, iso_theo)
                    n_iso = int((obs > 0).sum())
                raw_ppm = (fpeak['mz'] - theo_mz) / theo_mz * 1e6
                rows.append({
                    'condition': condition, 'run': run, 'level': _PANEL_LEVELS[1],
                    'label': label, 'sequence': seq, 'charge': fz,
                    'mz_theoretical': theo_mz, 'mz_observed': fpeak['mz'],
                    'ppm': raw_ppm - off, 'ppm_raw': raw_ppm, 'ppm_cal': off,
                    'fwhm': fpeak['fwhm'],
                    'resolution': fpeak['mz'] / fpeak['fwhm'],
                    'height': fpeak['height'], 'area': fpeak['area'],
                    'sn': fpeak['sn'], 'feature_intensity': t.intensity,
                    'profile': bool(np.isfinite(fpeak['fwhm'])),
                    'rt': apex_rt, 'rt_engine': float(t.rt),
                    'rt_left': lo_rt, 'rt_right': hi_rt,
                    'n_coeluting': n_coelute, 'n_fragments': len(measured),
                    'precursor_mz': float(t.mz_theoretical),
                    'precursor_charge': z,
                    'isotope_cosine': cos, 'n_isotopes_observed': n_iso,
                })

            in_peak = np.nonzero((idx.ms1_rt >= lo_rt) & (idx.ms1_rt <= hi_rt))[0]
            if in_peak.size == 0:
                in_peak = np.array([int(np.argmin(np.abs(idx.ms1_rt - apex_rt)))])
            if in_peak.size > 1:
                smz, sin = _sum_spectra([idx.spectrum(idx.ms1_pos[i])
                                         for i in in_peak])
            else:
                smz, sin = idx.spectrum(idx.ms1_pos[in_peak[0]])

            off1 = float(cal1(t.mz_theoretical)) if cal1 is not None else 0.0
            peak = measure_peak(smz, sin, t.mz_theoretical, profile=True,
                                tol_ppm=tol_ppm, offset_ppm=off1)
            if peak is None:
                continue
            iso_theo = theoretical_isotopes(_residue_formula(seq, mods)
                                            + [('H', 2), ('O', 1)])
            cos, n_iso = np.nan, 0
            if iso_theo is not None:
                obs = measure_isotopes(smz, sin, t.mz_theoretical, z,
                                       len(iso_theo), profile=True,
                                       tol_ppm=tol_ppm, calib=cal1)
                cos = cosine_similarity(obs, iso_theo)
                n_iso = int((obs > 0).sum())
            raw_ppm = (peak['mz'] - t.mz_theoretical) / t.mz_theoretical * 1e6
            rows.append({
                'condition': condition, 'run': run, 'level': _PANEL_LEVELS[0],
                'label': 'precursor', 'sequence': seq, 'charge': z,
                'mz_theoretical': t.mz_theoretical, 'mz_observed': peak['mz'],
                'ppm': raw_ppm - off1, 'ppm_raw': raw_ppm, 'ppm_cal': off1,
                'fwhm': peak['fwhm'],
                'resolution': peak['mz'] / peak['fwhm'],
                'height': peak['height'], 'area': peak['area'], 'sn': peak['sn'],
                'feature_intensity': t.intensity,
                'profile': bool(np.isfinite(peak['fwhm'])),
                'rt': apex_rt, 'rt_engine': float(t.rt),
                    'rt_left': lo_rt, 'rt_right': hi_rt,
                'n_coeluting': n_coelute, 'n_fragments': len(measured),
                'n_ms1_scans': int(in_peak.size),
                'isotope_cosine': cos, 'n_isotopes_observed': n_iso,
            })

            if iso_theo is None or not MEASURE_MS1_ISOTOPES:
                continue
            for k in range(1, len(iso_theo)):
                iso_mz = t.mz_theoretical + k * ISOTOPE_SPACING / z
                off_k = float(cal1(iso_mz)) if cal1 is not None else 0.0
                ipk = measure_peak(smz, sin, iso_mz, profile=True,
                                   tol_ppm=tol_ppm, offset_ppm=off_k)
                if ipk is None:
                    continue
                raw_ppm = (ipk['mz'] - iso_mz) / iso_mz * 1e6
                rows.append({
                    'condition': condition, 'run': run,
                    'level': _PANEL_LEVELS[0], 'label': f'M+{k}',
                    'sequence': seq, 'charge': z,
                    'mz_theoretical': iso_mz, 'mz_observed': ipk['mz'],
                    'ppm': raw_ppm - off_k, 'ppm_raw': raw_ppm,
                    'ppm_cal': off_k, 'fwhm': ipk['fwhm'],
                    'resolution': ipk['mz'] / ipk['fwhm'],
                    'height': ipk['height'], 'area': ipk['area'],
                    'sn': ipk['sn'], 'feature_intensity': t.intensity,
                    'profile': bool(np.isfinite(ipk['fwhm'])),
                    'rt': apex_rt, 'rt_engine': float(t.rt),
                    'rt_left': lo_rt, 'rt_right': hi_rt,
                    'n_coeluting': n_coelute, 'n_fragments': len(measured),
                    'n_ms1_scans': int(in_peak.size),
                    'isotope_cosine': np.nan, 'n_isotopes_observed': n_iso,
                })
        return rows

    cals = {}
    if CALIBRATE:
        sub = calibration_subset(targets)
        if verbose:
            print(f'    calibration pass: {len(sub):,} targets at '
                  f'+-{CALIB_SEARCH_PPM:g} ppm')
        cals = _fit_calibrations(measure(sub, {}, CALIB_SEARCH_PPM),
                                 verbose=verbose)

    tol = CALIB_TOL_PPM if (CALIBRATE and cals) else SEARCH_PPM
    if verbose:
        print(f'    measurement pass: {len(targets):,} targets at +-{tol:g} ppm'
              f'{" of the calibrated position" if cals else ""}')
    out = pd.DataFrame(measure(targets, cals, tol))

    del ms, idx
    gc.collect()
    return out


def _raw_for_run(run):
    """Locate the raw file for a run name, anywhere under the data root or the known sibling raw folders."""
    roots = [tqc.DATA_ROOT, *EXTRA_RAW_DIRS,
             os.path.dirname(os.path.abspath(tqc.DATA_ROOT))]
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for ext in ('.wiff', '.raw'):
                cand = os.path.join(dirpath, run + ext)
                if os.path.isfile(cand):
                    return cand
    return None


EXTRA_RAW_DIRS = []


def run_extraction(verbose=True):
    """Extract every condition's chosen run(s); returns the combined table."""
    if tqc.CONDITIONS is None:
        configure()
    cache = _profile_cache_dir()
    os.makedirs(cache, exist_ok=True)

    parts = []
    for cond in tqc.COND_ORDER:
        path = tqc.condition_path(cond)
        if not path.lower().endswith('.csv'):
            print(f'  {cond}: not a PEAKS features.csv, skipped')
            continue
        header = pd.read_csv(path, nrows=0).columns
        runs = [c[:-len(' m/z')] for c in header
                if c.endswith(' m/z') and c != 'm/z']
        only = tqc.condition_runs(cond)
        if only is not None:
            runs = [r for r in runs if r in set(only)]
        runs = runs[:RUNS_PER_CONDITION]
        for run in runs:
            out_path = os.path.join(cache, f'{run}.rawdiag.parquet')
            if os.path.exists(out_path) and not FORCE:
                cached = pd.read_parquet(out_path)
                was = cached['condition'].iloc[0] if len(cached) else cond
                if was != cond:
                    print(f'  {cond}/{run}: cached under {was!r}, re-labelled')
                cached['condition'] = cond
                parts.append(cached)
                print(f'  {cond}/{run}: reused {os.path.basename(out_path)}')
                continue
            if not USE_RAW:
                print(f'  {cond}/{run}: no cached measurements and USE_RAW is off')
                continue
            raw = _raw_for_run(run)
            if raw is None:
                print(f'  {cond}/{run}: raw file not found, skipped')
                continue
            targets = sample_precursors(precursor_table(path, run))
            print(f'  {cond}/{run}: {len(targets):,} sampled precursors -> {raw}')
            res = extract_run(raw, targets, condition=cond, run=run, verbose=verbose)
            res.to_parquet(out_path, index=False)
            print(f'    measured {len(res):,} peaks -> {out_path}')
            parts.append(res)
    if not parts:
        raise RuntimeError('no measurements; run with --use-raw first')
    return pd.concat(parts, ignore_index=True)


def _robust_sd(x):
    """IQR/1.349 - see ``core.sigma_iqr``, which this now defers to."""
    return sigma_iqr(x)


def _hist_on_ax(ax, d, value_col, *, level, xlabel, title, rows, panel,
                unit='', xlim=None, note_missing=None, coverage=0.99,
                min_counts=250, bins_range=(20, 100), center=False,
                show_ci=False):
    """House distribution rendering: filled frequency polygon over binned counts, one curve per condition, thin per-run curves behind."""
    from lc_peak_width_analysis import _n_bins

    sub = d[(d['level'] == level) & d[value_col].notna()].copy()
    conds = [c for c in tqc.COND_ORDER if (sub['condition'] == c).any()]
    sub = sub[sub['condition'].isin(conds)]

    offsets = {}
    if center and conds:
        offsets = {c: float(sub.loc[sub['condition'] == c, value_col].median())
                   for c in conds}
        sub[value_col] = sub[value_col] - sub['condition'].map(offsets)
    missing = [c for c in tqc.COND_ORDER
               if (d['condition'] == c).any() and c not in conds]

    if not conds:
        ax.text(0.5, 0.5, note_missing or 'no data', ha='center', va='center',
                fontsize=10, color='#666666', transform=ax.transAxes)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['top', 'right']].set_visible(False)
        return

    vals = sub[value_col]
    if xlim is None:
        q = (1 - coverage) / 2
        lo, hi = np.quantile(vals, [q, 1 - q])
        pad = 0.05 * (hi - lo)
        lo, hi = lo - pad, hi + pad
    else:
        lo, hi = xlim
    inside = vals[vals.between(lo, hi)]
    smallest = min(int(sub.loc[sub['condition'] == c, value_col]
                       .between(lo, hi).sum()) for c in conds)
    n_bins = int(np.clip(round(min(_n_bins(inside, lo, hi, 10 ** 9),
                                   max(smallest, 1) / min_counts)),
                         *bins_range))
    edges = np.linspace(lo, hi, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    bin_w = edges[1] - edges[0]

    labels = []
    for cond in conds:
        g = sub[sub['condition'] == cond]
        color = tqc.COLOR[cond]
        for run, gr in g.groupby('run', sort=False):
            c, _ = np.histogram(gr[value_col], bins=edges)
            n_r = max(int(gr[value_col].notna().sum()), 1)
            ax.plot(centres, c / n_r, color=color, lw=0.6, alpha=0.35, zorder=2)
        counts, _ = np.histogram(g[value_col], bins=edges)
        counts = counts / max(len(g), 1)
        ax.plot(centres, counts, color=color, lw=2.2, zorder=3)
        ax.fill_between(centres, counts, color=color, alpha=0.18, zorder=1)
        med = float(g[value_col].median())
        ax.axvline(med, color=color, ls='--', lw=1.0, alpha=0.9, zorder=4)
        rsd = _robust_sd(g[value_col])
        off_c = float((~g[value_col].between(lo, hi)).mean() * 100)
        p95 = float(np.quantile(np.abs(g[value_col] - med), 0.95))
        fmt = ',.2f' if abs(rsd) < 1000 else ',.0f'
        head = (f'{offsets[cond]:{fmt}} →' if center else f'{med:{fmt}}')
        labels.append(f'{cond}  {head} ± {rsd:{fmt}}'
                      + (f'   ±{p95:{fmt}}' if show_ci else '')
                      + f'   {off_c:.1f}%')
        rows.append(pd.DataFrame({
            'panel': panel, 'level': level, 'condition': cond,
            'x': centres, 'fraction': counts, 'median': med,
            'offset_removed': offsets.get(cond, 0.0),
            'robust_sd': rsd, 'p95_abs_dev': p95, 'pct_off_axis': off_c,
            'n': len(g)}))

    if missing:
        ax.text(0.02, 0.98, '\n'.join(f'{c}: {note_missing}' for c in missing),
                transform=ax.transAxes, va='top', ha='left', fontsize=8.5,
                color='#666666')
    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(f'Fraction of peaks (bin {bin_w:,.3g})', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold',
                 pad=14 + 12.5 * (len(conds) + 2))
    ax.spines[['top', 'right']].set_visible(False)
    key = ('offset removed → ± σ_IQR' if center else 'median ± σ_IQR')
    key += '   (IQR/1.349)\n' + ('95% of |dev| · ' if show_ci else '') + '% off axis'
    tqc._compact_legend(ax, conds, labels=labels, loc='lower center',
                        bbox_to_anchor=(0.5, 1.005), fontsize=9.5,
                        title=key + (f'   [{unit.strip()}]' if unit.strip() else ''),
                        title_fontsize=9.5, alpha=0.18)


def _vs_x_on_ax(ax, d, value_col, x_col, *, xlabel, ylabel, title, rows, panel,
                points_per_bin=200, bins_range=(5, 25), log_x=False,
                zero_line=False):
    """Median of `value_col` against `x_col`, both levels on one axes."""
    drawn = []
    for level, style in zip(_PANEL_LEVELS, ('-', '--')):
        sub = d[(d['level'] == level) & d[value_col].notna()
                & d[x_col].notna() & (d[x_col] > 0)]
        for cond in tqc.COND_ORDER:
            g = sub[sub['condition'] == cond]
            n_bins = int(np.clip(round(len(g) / points_per_bin), *bins_range))
            if len(g) < bins_range[0] * 5:
                continue
            b = pd.qcut(g[x_col], n_bins, labels=False, duplicates='drop')
            agg = g.groupby(b, observed=True).agg(
                x=(x_col, lambda s: np.log10(np.median(s)) if log_x
                   else float(np.median(s))),
                y=(value_col, 'median'))
            ax.plot(agg['x'], agg['y'], color=tqc.COLOR[cond], lw=1.8, ls=style,
                    marker='o', ms=3.0, zorder=3)
            drawn.append(cond)
            rows.append(pd.DataFrame({
                'panel': panel, 'level': level, 'condition': cond,
                'x': agg['x'].values, 'median': agg['y'].values,
                'n_bins': n_bins, 'points_per_bin': len(g) / max(n_bins, 1)}))
    if zero_line:
        ax.axhline(0, color='black', lw=0.8, zorder=2)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    order = [c for c in tqc.COND_ORDER if c in set(drawn)]
    if order:
        tqc._compact_legend(ax, order, loc='best', fontsize=9.5,
                            title=f'solid = MS1, dashed = MS2\n'
                                  f'~{points_per_bin} peaks per bin',
                            title_fontsize=9.5)


def plot_metric(d, value_col, *, name, xlabel, ylabel, what, unit='',
                xlim=None, note_missing=None, zero_line=False,
                coverage=0.99, min_counts=250, bins_range=(20, 100),
                center=False, show_ci=False):
    """One 2x2 per metric: the distribution at each level, then the metric against m/z and against abundance with both levels overlaid."""
    rows = []
    n_conds = d['condition'].nunique()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8 + 0.26 * (n_conds + 2)))
    for ax, level in zip(axes[0], _PANEL_LEVELS):
        _hist_on_ax(ax, d, value_col, level=level, rows=rows, panel=name,
                    xlabel=xlabel, unit=unit, xlim=xlim,
                    note_missing=note_missing, coverage=coverage,
                    min_counts=min_counts, bins_range=bins_range,
                    center=center, show_ci=show_ci,
                    title=f'{what}\n({level})')
    _vs_x_on_ax(axes[1, 0], d, value_col, 'mz_theoretical', rows=rows,
                panel=f'{name}_vs_mz', xlabel='m/z (theoretical)',
                ylabel=f'Median {ylabel}', title=f'{what} vs m/z',
                zero_line=zero_line)
    _vs_x_on_ax(axes[1, 1], d, value_col, 'feature_intensity', rows=rows,
                panel=f'{name}_vs_abundance', log_x=True,
                xlabel='log$_{10}$ precursor intensity',
                ylabel=f'Median {ylabel}', title=f'{what} vs abundance',
                zero_line=zero_line)
    for ax in axes.ravel():
        tqc._unbold_axis_labels(ax)
    if TOP_INTENSITY_FRAC and TOP_INTENSITY_FRAC < 1:
        fig.suptitle(f'top {TOP_INTENSITY_FRAC:.0%} by abundance  ·  '
                     f'±{CALIB_TOL_PPM:g} ppm after calibration',
                     fontsize=9, y=0.995)
    fig.tight_layout()
    save(fig, name, pd.concat(rows, ignore_index=True))


def plot_mass_accuracy(d):
    plot_metric(d, 'ppm', name='raw_mass_accuracy',
                what='Mass accuracy (calibrated)',
                xlabel='Calibrated mass error [ppm]',
                ylabel='mass error [ppm]', unit=' ppm', zero_line=True,
                coverage=0.98, min_counts=80, bins_range=(30, 110),
                center=True, show_ci=True)


def report_calibration(d):
    """What the calibration removed, and what it left."""
    if 'ppm_cal' not in d.columns:
        return
    print('\ncalibration (per condition and level):')
    print(f"  {'condition':>12} {'level':>14} {'n':>8} {'removed':>18} "
          f"{'residual sigma_IQR':>20} {'95% |resid|':>13} {'|resid|>tol':>12}")
    for (cond, level), g in d.groupby(['condition', 'level']):
        cal = g['ppm_cal'].dropna()
        r = g['ppm'].dropna().to_numpy()
        if not len(r):
            continue
        sig = (np.quantile(r, .75) - np.quantile(r, .25)) / 1.349
        p95 = np.quantile(np.abs(r), 0.95)
        over = (np.abs(r) > CALIB_TOL_PPM).mean() * 100
        span = (f'{cal.min():+.2f} to {cal.max():+.2f}' if len(cal)
                else 'none')
        print(f'  {cond:>12} {level:>14} {len(r):>8,} {span:>18} '
              f'{sig:>20.3f} {p95:>13.2f} {over:>11.1f}%')


def plot_resolution(d):
    plot_metric(d, 'resolution', name='raw_resolution',
                what='m/z resolution', xlabel='Resolution (m/z ÷ FWHM)',
                ylabel='resolution', coverage=0.90, min_counts=40,
                bins_range=(30, 120),
                note_missing='centroid data — no peak shape')


def plot_isotopes(d):
    plot_metric(d, 'isotope_cosine', name='raw_isotopes',
                what='Isotope envelope', xlim=(0, 1.02),
                xlabel='Cosine similarity to theoretical envelope',
                ylabel='cosine similarity', min_counts=40,
                bins_range=(40, 120))


EXAMPLE_QUARTILES = ('lowest', 'low', 'medium', 'high')
THEORETICAL_COLOR = '#333333'
EXAMPLE_ISO_HALFWIDTH = 0.04
XIC_PLOT_WINDOW = 1.0
N_XIC_FRAGMENTS = 5
FULL_MS2_MIN_REL = 0.001
MATCH_TOL_PPM = 20.0

PEPTDEEP_SCIEX_MODEL = r'Z:\Search_settings\peptdeep\8600\peptdeep.transfer'
PEPTDEEP_MODEL_BY_CONDITION = {}
PEPTDEEP_NCE = 30.0
PEPTDEEP_INSTRUMENT = 'Lumos'
_PEPTDEEP_CACHE = {}


def predict_ms2(seq, charge, *, model_dir=None, nce=None, instrument=None):
    """PeptDeep-predicted fragment m/z and relative intensities."""
    from peptdeep.pretrained_models import ModelManager

    key = model_dir or '_builtin_'
    mgr = _PEPTDEEP_CACHE.get(key)
    if mgr is None:
        mgr = ModelManager(mask_modloss=True, device='cpu')
        mgr.load_installed_models()
        if model_dir:
            mgr.load_external_models(
                ms2_model_file=os.path.join(model_dir, 'ms2.pth'),
                rt_model_file=os.path.join(model_dir, 'rt.pth'),
                ccs_model_file=os.path.join(model_dir, 'ccs.pth'))
        _PEPTDEEP_CACHE[key] = mgr

    df = pd.DataFrame({
        'sequence': [str(seq)], 'mods': [''], 'mod_sites': [''],
        'charge': [int(charge)],
        'nce': [PEPTDEEP_NCE if nce is None else nce],
        'instrument': [PEPTDEEP_INSTRUMENT if instrument is None else instrument],
    })
    out = mgr.predict_all(df, predict_items=['ms2'])
    mz_df = out['fragment_mz_df']
    int_df = out['fragment_intensity_df']
    if mz_df is None or len(mz_df) == 0:
        return None

    labels, mzs, ints = [], [], []
    n = len(str(seq))
    for col in mz_df.columns:
        m = mz_df[col].values
        v = int_df[col].values if col in int_df.columns else np.zeros_like(m)
        ion = col.split('_')[0]
        fz = int(col.split('z')[-1]) if 'z' in col else 1
        for i, (mm, vv) in enumerate(zip(m, v)):
            if mm <= 0 or vv <= 0:
                continue
            idx = i + 1 if ion.startswith('b') else n - i - 1
            labels.append(f'{ion}{idx}' + ('' if fz == 1 else f'^{fz}'))
            mzs.append(float(mm))
            ints.append(float(vv))
    if not mzs:
        return None
    ints = np.asarray(ints)
    return labels, np.asarray(mzs), ints / ints.max()


def zero_fill(mz, inten, *, gap_factor=3.0):
    """Insert explicit zeros across the gaps between stored points."""
    mz = np.asarray(mz, dtype=float)
    inten = np.asarray(inten, dtype=float)
    if mz.size < 2:
        return mz, inten
    d = np.diff(mz)
    step = float(np.median(d[d > 0])) if np.any(d > 0) else 1e-4
    pad = max(step * 0.5, 1e-6)
    gap = d > max(gap_factor * step, 2e-6)

    out_mz = [mz[0] - pad]
    out_in = [0.0]
    for i in range(mz.size):
        out_mz.append(mz[i])
        out_in.append(inten[i])
        if i < gap.size and gap[i]:
            out_mz += [mz[i] + pad, mz[i + 1] - pad]
            out_in += [0.0, 0.0]
    out_mz.append(mz[-1] + pad)
    out_in.append(0.0)
    return np.asarray(out_mz), np.asarray(out_in)
EXAMPLE_HALFWIDTH = {'MS1 precursor': 2.6, 'MS2 fragment': 1.6}


def extract_examples(raw_path, measured, *, condition, run, verbose=True):
    """Profile slices for one representative peak per intensity quartile."""
    from zenotof8600_analytics.raw import load_ms_data
    from zenotof8600_analytics.raw.extraction import peak_from_fragments

    ms = load_ms_data(raw_path, cache_dir=_profile_cache_dir(), centroided=False,
                      verbose=verbose)
    sdf = ms.spectrum_df
    mz_all = ms.peak_df['mz'].values
    in_all = ms.peak_df['intensity'].values
    ms1 = sdf[sdf['ms_level'] == 1].sort_values('rt')
    ms1_rt, ms1_idx = ms1['rt'].values, ms1.index.values
    _starts = sdf['peak_start_idx'].to_numpy(np.int64)
    _stops = sdf['peak_stop_idx'].to_numpy(np.int64)
    windows = _ms2_scan_index(sdf)

    measured = measured.copy()
    if 'precursor_mz' in measured.columns:
        measured['_prec_mz'] = measured['precursor_mz']
    else:
        ms1_rows = measured[measured['level'] == _PANEL_LEVELS[0]] \
            .drop_duplicates(['sequence', 'rt'])
        lut = dict(zip(zip(ms1_rows['sequence'], ms1_rows['rt']),
                       ms1_rows['mz_theoretical']))
        measured['_prec_mz'] = [lut.get((s, r), np.nan) for s, r
                                in zip(measured['sequence'], measured['rt'])]

    out = []
    for level in _PANEL_LEVELS:
        sub = measured[(measured['level'] == level)
                       & measured['feature_intensity'].notna()
                       & measured['rt'].notna()].copy()
        if level == _PANEL_LEVELS[1]:
            sub = sub[sub['_prec_mz'].notna()]
        else:
            sub = sub[sub['label'] == 'precursor']
        if sub.empty:
            continue
        sub['_q'] = pd.qcut(sub['feature_intensity'], 4, labels=False,
                            duplicates='drop')
        half = EXAMPLE_HALFWIDTH[level]
        for q, g in sub.groupby('_q'):
            if level == _PANEL_LEVELS[0]:
                g = g.sort_values(['n_isotopes_observed', 'isotope_cosine'],
                                  ascending=False)
                row = g.iloc[0]
            else:
                g = g.sort_values('sn')
                row = g.iloc[len(g) // 2]
            lo_rt = float(row['rt_left']) if 'rt_left' in row else np.nan
            hi_rt = float(row['rt_right']) if 'rt_right' in row else np.nan
            if not (np.isfinite(lo_rt) and np.isfinite(hi_rt) and hi_rt > lo_rt):
                lo_rt = float(row['rt']) - RT_WINDOW
                hi_rt = float(row['rt']) + RT_WINDOW
            if level == _PANEL_LEVELS[0]:
                sel = np.nonzero((ms1_rt >= lo_rt) & (ms1_rt <= hi_rt))[0]
                if sel.size == 0:
                    sel = np.array([int(np.argmin(np.abs(ms1_rt - row['rt'])))])
                idxs = [int(ms1_idx[i]) for i in sel]
            else:
                centre_rt = 0.5 * (lo_rt + hi_rt)
                cands = _candidate_ms2_scans(windows, row['_prec_mz'], centre_rt,
                                             max(0.5 * (hi_rt - lo_rt), 1e-6))
                if not cands:
                    cands = _candidate_ms2_scans(windows, row['_prec_mz'],
                                                 float(row['rt']), RT_WINDOW)
                if not cands:
                    continue
                idxs = [int(i) for _, i in cands]
            sidx = idxs[0]
            smz, sin = (_sum_spectra([(mz_all[_starts[i]:_stops[i]],
                                       in_all[_starts[i]:_stops[i]])
                                      for i in idxs])
                        if len(idxs) > 1
                        else (mz_all[_starts[sidx]:_stops[sidx]],
                              in_all[_starts[sidx]:_stops[sidx]]))
            srow = sdf.loc[sidx]
            centre = float(row['mz_theoretical'])
            z = max(int(row['charge']), 1)

            iso = None
            try:
                seq = str(row['sequence'])
                extra = [('H', 2), ('O', 1)] if level == _PANEL_LEVELS[0] else []
                if str(row['label']).startswith('y'):
                    extra = [('H', 2), ('O', 1)]
                iso = theoretical_isotopes(_residue_formula(seq, []) + extra)
            except Exception:
                iso = None

            n_iso = len(iso) if iso is not None else 4
            lo_mz = centre - 0.35 / z
            hi_mz = centre + (n_iso - 0.5) / z
            lo, hi = np.searchsorted(smz, [lo_mz, hi_mz])
            if hi - lo < 3:
                continue
            if level == _PANEL_LEVELS[1] and np.isfinite(row.get('_prec_mz', np.nan)):
                try:
                    frg = fragment_targets(str(row['sequence']), [],
                                           float(row['_prec_mz']),
                                           int(row.get('precursor_charge', 2) or 2))
                    if frg:
                        fmzs = [f[1] for f in frg[:PEAK_PICK_FRAGMENTS]]
                        _pk_wide, trace, _u = peak_from_fragments(
                            ms, float(row['_prec_mz']), fmzs,
                            rt_range=(row['rt'] - XIC_PLOT_WINDOW,
                                      row['rt'] + XIC_PLOT_WINDOW),
                            mz_tol=SEARCH_PPM, top_n=6, baseline='p5')
                        pk = peak_from_fragments(
                            ms, float(row['_prec_mz']), fmzs,
                            rt_range=(row['rt'] - RT_WINDOW,
                                      row['rt'] + RT_WINDOW),
                            mz_tol=SEARCH_PPM, top_n=6, baseline='p5')[0]
                        from zenotof8600_analytics.raw.extraction import extract_xic
                        per_frag = []
                        for lbl, fmz in [(f[0], f[1]) for f in
                                         frg[:PEAK_PICK_FRAGMENTS]]:
                            x = extract_xic(
                                ms, float(fmz), mz_tol=SEARCH_PPM,
                                mz_tol_unit='ppm',
                                rt_range=(row['rt'] - XIC_PLOT_WINDOW,
                                          row['rt'] + XIC_PLOT_WINDOW),
                                ms_level=2, precursor_mz=float(row['_prec_mz']))
                            if len(x):
                                per_frag.append((lbl, fmz,
                                                 float(x['intensity'].sum()), x))
                        per_frag.sort(key=lambda r: -r[2])
                        for lbl, fmz, _tot, x in per_frag[:N_XIC_FRAGMENTS]:
                            out.append(pd.DataFrame({
                                'condition': condition, 'run': run,
                                'level': level,
                                'quartile': EXAMPLE_QUARTILES[int(q)],
                                'sequence': row['sequence'], 'label': lbl,
                                'charge': z, 'target_mz': float(fmz),
                                'rt': float(row['rt']),
                                'feature_intensity': float(row['feature_intensity']),
                                'kind': 'xic_frag',
                                'mz': x['rt'].values,
                                'intensity': x['intensity'].values,
                                'apex_rt': pk.get('apex_rt', np.nan),
                                'left_rt': pk.get('left_rt', np.nan),
                                'right_rt': pk.get('right_rt', np.nan)}))
                        if len(trace):
                            out.append(pd.DataFrame({
                                'condition': condition, 'run': run,
                                'level': level,
                                'quartile': EXAMPLE_QUARTILES[int(q)],
                                'sequence': row['sequence'],
                                'label': row['label'],
                                'charge': z, 'target_mz': centre,
                                'rt': float(row['rt']),
                                'feature_intensity': float(row['feature_intensity']),
                                'kind': 'xic',
                                'mz': trace['rt'].values,
                                'intensity': trace['intensity'].values,
                                'apex_rt': pk.get('apex_rt', np.nan),
                                'left_rt': pk.get('left_rt', np.nan),
                                'right_rt': pk.get('right_rt', np.nan)}))
                except Exception:
                    pass

            common = {
                'condition': condition, 'run': run, 'level': level,
                'quartile': EXAMPLE_QUARTILES[int(q)],
                'sequence': row['sequence'], 'label': row['label'],
                'charge': z, 'target_mz': centre, 'rt': float(srow['rt']),
                'feature_intensity': float(row['feature_intensity']),
                'fwhm': float(row['fwhm']) if np.isfinite(row['fwhm']) else np.nan,
                'n_scans': len(idxs),
                'iso_cosine': float(row['isotope_cosine'])
                if np.isfinite(row['isotope_cosine']) else np.nan,
                'ppm_row': float(row['ppm']) if np.isfinite(row['ppm']) else np.nan,
                'ppm_cal': float(row['ppm_cal'])
                if np.isfinite(row.get('ppm_cal', np.nan)) else np.nan,
            }
            out.append(pd.DataFrame({**common, 'kind': 'experimental',
                                     'mz': smz[lo:hi], 'intensity': sin[lo:hi]}))
            if iso is not None:
                scale = float(np.max(sin[lo:hi])) / float(np.max(iso))
                out.append(pd.DataFrame({
                    **common, 'kind': 'theoretical',
                    'mz': centre + np.arange(len(iso)) * ISOTOPE_SPACING / z,
                    'intensity': np.asarray(iso) * scale}))

            if level == _PANEL_LEVELS[1] and EXAMPLE_QUARTILES[int(q)] == 'high':
                thr = float(np.max(sin)) * FULL_MS2_MIN_REL
                m = sin > thr
                out.append(pd.DataFrame({
                    **common, 'kind': 'ms2_full',
                    'mz': smz[m], 'intensity': sin[m]}))
    del ms, mz_all, in_all
    gc.collect()
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def plot_examples(ex, level, condition):
    """2x2 of raw spectra for one condition, one intensity quartile per panel."""
    sub = ex[(ex['level'] == level) & (ex['condition'] == condition)]
    if sub.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    colour = tqc.COLOR.get(condition, tqc.PALETTE[0])
    rows = []
    for ax, q in zip(axes.ravel(), EXAMPLE_QUARTILES):
        g = sub[sub['quartile'] == q]
        obs = g[g['kind'] == 'experimental']
        if obs.empty:
            ax.set_visible(False)
            continue
        key = obs.iloc[0]
        z = max(int(key['charge']), 1)

        theo_pos = (g.loc[g['kind'] == 'theoretical', 'mz'].values
                    if (g['kind'] == 'theoretical').any()
                    else key['target_mz'] + np.arange(4) * ISOTOPE_SPACING / z)
        halfw = min(EXAMPLE_ISO_HALFWIDTH, 0.3 / z)
        omz, oin = obs['mz'].values, obs['intensity'].values
        keep = np.zeros(omz.size, dtype=bool)
        for t in theo_pos:
            keep |= np.abs(omz - t) <= halfw
        if keep.sum() < 3:
            keep = np.ones(omz.size, dtype=bool)
        mz, inten = zero_fill(omz[keep], oin[keep])
        ax.plot(mz, inten, color=colour, lw=1.1, zorder=3, label='measured')
        ax.fill_between(mz, inten, color=colour, alpha=0.25, zorder=2)

        theo = g[g['kind'] == 'theoretical']
        if not theo.empty:
            tmz = theo['mz'].values
            trel = theo['intensity'].values.astype(float)
            trel = trel / trel.max() if trel.max() > 0 else trel
            fwhm0 = float(key['fwhm']) if 'fwhm' in key else np.nan
            cal_off = 0.0
            if 'ppm_cal' in key and np.isfinite(key['ppm_cal']):
                cal_off = float(key['ppm_cal'])
            mtol = np.maximum(0.5 * (fwhm0 if np.isfinite(fwhm0) else 0.0),
                              tmz * CALIB_TOL_PPM * 1e-6)
            meas_h = np.array([oin[np.abs(omz - t) <= w].max()
                               if np.any(np.abs(omz - t) <= w) else 0.0
                               for t, w in zip(tmz, mtol)])
            denom = float((trel * trel).sum())
            scale = (float((meas_h * trel).sum()) / denom if denom > 0 else 0.0)
            if not (scale > 0):
                scale = float(meas_h.max()) or 1.0
            tint = trel * scale
            fwhm = fwhm0
            if not (np.isfinite(fwhm) and fwhm > 0):
                fwhm = np.nan
            if np.isfinite(fwhm):
                sigma = fwhm / 2.35482
                x0, x1 = float(mz.min()), float(mz.max())
                step = max(fwhm / 25.0, (x1 - x0) / 20000.0)
                grid = np.arange(x0, x1 + step, step)
                prof = np.zeros_like(grid)
                for tt, aa in zip(tmz, tint):
                    prof += aa * np.exp(-0.5 * ((grid - tt) / sigma) ** 2)
                ax.plot(grid, prof, color=THEORETICAL_COLOR, lw=1.4, zorder=5,
                        label='theoretical')
            else:
                ax.vlines(tmz, 0, tint, color=THEORETICAL_COLOR, lw=1.6,
                          zorder=5, label='theoretical')
                ax.scatter(tmz, tint, s=16, color=THEORETICAL_COLOR, zorder=6)
            top = max(float(meas_h.max()), float(tint.max()))
            ax.set_ylim(0, 1.15 * top)
            for tt, aa, mm in zip(tmz, tint, meas_h):
                if mm <= 0 or aa <= 0:
                    continue
                seg = np.abs(omz - tt) <= max(0.5 * fwhm0, tt * CALIB_TOL_PPM * 1e-6) \
                    if np.isfinite(fwhm0) else np.abs(omz - tt) <= tt * CALIB_TOL_PPM * 1e-6
                if not seg.any():
                    continue
                obs_mz = float(omz[seg][int(np.argmax(oin[seg]))])
                dev = (obs_mz - tt) / tt * 1e6 - cal_off
                ax.annotate(f'{dev:+.1f} ppm\n'
                            f'{mm / aa:.2f}\u00d7',
                            xy=(tt, max(mm, aa)), xytext=(0, 3),
                            textcoords='offset points', ha='center', va='bottom',
                            fontsize=8, color='0.25', linespacing=1.1)
            ax.set_ylim(0, 1.42 * top)
        ax.axhline(0, color='black', lw=0.8, zorder=1)

        seq = str(key['sequence'])
        ax.set_xlabel('m/z', fontsize=10)
        ax.set_ylabel('Intensity', fontsize=10)
        cos = float(key['iso_cosine']) if 'iso_cosine' in key else np.nan
        ax.set_title(f'{q} — {seq[:12]}{"…" if len(seq) > 12 else ""} {z}+'
                     + (f'\ncos {cos:.3f}' if np.isfinite(cos) else ''),
                     fontsize=10, fontweight='bold')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        span = float(np.ptp(ax.get_xlim()))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(
            '%.2f' if span < 2 else '%.1f'))
        ax.spines[['top', 'right']].set_visible(False)
        if not rows:
            ax.legend(loc='upper right', fontsize=9, frameon=False,
                      handlelength=1.2, labelspacing=0.3)
        tqc._unbold_axis_labels(ax)
        rows.append(g)
    if not rows:
        plt.close(fig)
        return
    fig.suptitle(f'{level} — {condition}', fontsize=11, fontweight='bold')
    fig.tight_layout()
    tag = 'ms1' if level == _PANEL_LEVELS[0] else 'ms2'
    save(fig, f'raw_example_spectra_{tag}_{condition}',
         pd.concat(rows, ignore_index=True), subdir=SPECTRA_SUBDIR)


def _match_alpharaw(spec_mz, spec_int, targets, tol_ppm):
    """Match theoretical m/z against a spectrum using alpharaw's matcher."""
    from alpharaw.match.match_utils import match_profile_peaks

    spec_mz = np.ascontiguousarray(spec_mz, dtype=np.float64)
    spec_int = np.ascontiguousarray(spec_int, dtype=np.float64)
    q = np.ascontiguousarray(targets, dtype=np.float64)
    tols = q * tol_ppm * 1e-6
    first, last = match_profile_peaks(spec_mz, spec_int, q, tols)
    inten = np.zeros(q.size)
    mz = np.zeros(q.size)
    for i in range(q.size):
        a = int(first[i])
        if a < 0:
            continue
        b = max(int(last[i]), a + 1)
        seg = spec_int[a:b]
        if not seg.size or seg.max() <= 0:
            continue
        k = int(np.argmax(seg))
        inten[i] = float(seg[k])
        mz[i] = float(spec_mz[a + k])
    return inten, mz


def plot_ms2_full(ex, table, condition):
    """The whole fragment spectrum of one precursor, mirrored against the theoretical b/y ions broadened to that condition's median resolution."""
    g = ex[(ex['condition'] == condition) & (ex['kind'] == 'ms2_full')]
    if g.empty:
        return
    key = g.iloc[0]
    seq = str(key['sequence'])
    z = int(key.get('precursor_charge', 2) or 2)
    prec_mz = float(key.get('precursor_mz', key['target_mz']))

    res = table[(table['condition'] == condition)
                & (table['level'] == _PANEL_LEVELS[1])]['resolution'].dropna()
    r_med = float(res.median()) if len(res) else 30000.0

    pred, predicted = None, False
    try:
        pred = predict_ms2(seq, z,
                           model_dir=PEPTDEEP_MODEL_BY_CONDITION.get(condition))
        predicted = pred is not None
    except Exception as exc:
        print(f'  {condition}: peptdeep prediction unavailable ({exc})')
    if predicted:
        labels_all, targets, theo_rel = pred[0], pred[1], pred[2]
        frags = list(zip(labels_all, targets))
    else:
        frags = [(f[0], f[1]) for f in
                 fragment_targets(seq, [], prec_mz, z, above_precursor=False)]
        if not frags:
            return
        labels_all = [f[0] for f in frags]
        targets = np.array([f[1] for f in frags])
        theo_rel = np.ones(len(frags))
    gmz, gin = g['mz'].values, g['intensity'].values
    lo, hi = float(gmz.min()), float(gmz.max())

    matched_int, matched_mz = _match_alpharaw(gmz, gin, targets, MATCH_TOL_PPM)
    hit = matched_int > 0
    top = float(matched_int.max()) if hit.any() else float(gin.max() or 1.0)

    explained = float(matched_int.sum() / gin.sum() * 100) if gin.sum() else 0.0

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colour = tqc.COLOR.get(condition, tqc.PALETTE[0])
    ax.vlines(matched_mz[hit], 0, matched_int[hit], color=colour, lw=1.4, zorder=3)
    ax.scatter(matched_mz[hit], matched_int[hit], s=10, color=colour, zorder=4)
    ax.vlines(targets, 0, -theo_rel * top, color=THEORETICAL_COLOR, lw=1.2,
              alpha=0.85, zorder=3)

    order = np.argsort(-matched_int)
    for i, j in enumerate([k for k in order if hit[k]][:12]):
        ax.annotate(labels_all[j], (targets[j], -top * (1.06 + 0.10 * (i % 2))),
                    ha='center', va='top', fontsize=8, color=THEORETICAL_COLOR)

    ax.axhline(0, color='black', lw=0.8, zorder=1)
    ax.set_ylim(-1.35 * top, 1.15 * top)
    ax.set_xlim(lo, hi)
    ax.set_xlabel('m/z', fontsize=10)
    ax.set_ylabel('Intensity', fontsize=10)
    ax.set_title(f'{seq} {z}+ — {condition}\n'
                 f'{int(hit.sum())}/{len(targets)} ions, {explained:.0f}% of TIC',
                 fontsize=11, fontweight='bold')
    g = g.assign(**{'matched_only': True})
    ax.spines[['top', 'right']].set_visible(False)
    tqc._unbold_axis_labels(ax)
    fig.tight_layout()
    save(fig, f'raw_ms2_full_{condition}', g, subdir=SPECTRA_SUBDIR)


def plot_xic(ex, condition):
    """The fragment XIC each example's RT range was picked from."""
    sub = ex[(ex['condition'] == condition) & (ex['kind'] == 'xic')]
    sub_f = ex[(ex['condition'] == condition) & (ex['kind'] == 'xic_frag')]
    if sub.empty and sub_f.empty:
        return
    if sub.empty:
        sub = sub_f.drop_duplicates(['quartile'])
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    colour = tqc.COLOR.get(condition, tqc.PALETTE[0])
    rows = []
    for ax, q in zip(axes.ravel(), EXAMPLE_QUARTILES):
        g = sub[sub['quartile'] == q]
        if g.empty:
            ax.set_visible(False)
            continue
        key = g.iloc[0]
        frag = sub_f[sub_f['quartile'] == q]
        if not frag.empty:
            for i, (lbl, fg) in enumerate(frag.groupby('label', sort=False)):
                ax.plot(fg['mz'], fg['intensity'].values, lw=1.3, alpha=0.85,
                        zorder=3,
                        color=plt.cm.viridis(i / max(N_XIC_FRAGMENTS - 1, 1)),
                        label=str(lbl))
            ax.legend(loc='upper right', fontsize=7.5, frameon=False,
                      handlelength=1.0, labelspacing=0.25)
        else:
            ax.plot(g['mz'], g['intensity'], color=colour, lw=1.4, zorder=3)
            ax.fill_between(g['mz'], g['intensity'], color=colour, alpha=0.20,
                            zorder=2)
        ax.set_ylim(bottom=0)
        ax.axvline(key['rt'], color='black', ls='-', lw=1.0, zorder=4)
        ax.axvspan(key['rt'] - RT_WINDOW, key['rt'] + RT_WINDOW,
                   color='black', alpha=0.06, zorder=1)
        for v, style in ((key['apex_rt'], '--'), (key['left_rt'], ':'),
                         (key['right_rt'], ':')):
            if np.isfinite(v):
                ax.axvline(v, color=THEORETICAL_COLOR, ls=style, lw=1.1, zorder=4)
        ax.set_xlabel('RT [min]', fontsize=10)
        ax.set_ylabel('Fragment intensity'
                      if not frag.empty else 'Summed fragment intensity',
                      fontsize=10)
        ax.set_title(f'{q}  ({str(key["sequence"])[:16]}, {key["label"]})',
                     fontsize=10, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        tqc._unbold_axis_labels(ax)
        rows.append(pd.concat([g, frag]) if not frag.empty else g)
    if not rows:
        plt.close(fig)
        return
    fig.suptitle(f'Fragment XIC — {condition}', fontsize=11, fontweight='bold')
    fig.tight_layout()
    save(fig, f'raw_xic_{condition}', pd.concat(rows, ignore_index=True),
          subdir=SPECTRA_SUBDIR)


def build_examples(table):
    """Extract example spectra for EVERY condition, one raw file each."""
    cache = _profile_cache_dir()
    parts = []
    for cond in tqc.COND_ORDER:
        g = table[table['condition'] == cond]
        if g.empty:
            continue
        run = g['run'].iloc[0]
        out_path = os.path.join(cache, f'{run}.examples.parquet')
        if os.path.exists(out_path) and not FORCE:
            parts.append(pd.read_parquet(out_path))
            continue
        if not USE_RAW:
            continue
        raw = _raw_for_run(run)
        if raw is None:
            print(f'  {cond}: raw not found, no example spectra')
            continue
        ex = extract_examples(raw, g, condition=cond, run=run, verbose=False)
        if not ex.empty:
            ex.to_parquet(out_path, index=False)
            print(f'  example spectra: {cond}/{run}')
            parts.append(ex)
    if not parts:
        print('  no example spectra available (needs --use-raw once)')
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def main():
    table = run_extraction()
    print('\nmeasured peaks:')
    print(table.groupby(['condition', 'level']).size().to_string())
    report_calibration(table)
    plot_mass_accuracy(table)
    plot_resolution(table)
    plot_isotopes(table)
    ex = build_examples(table)
    if not ex.empty:
        for level in _PANEL_LEVELS:
            for cond in tqc.COND_ORDER:
                plot_examples(ex, level, cond)
        for cond in tqc.COND_ORDER:
            plot_ms2_full(ex, table, cond)
            plot_xic(ex, cond)
    print('\nper condition / level:')
    for (cond, lvl), g in table.groupby(['condition', 'level']):
        res = (f'{g["resolution"].median():,.0f}' if g['resolution'].notna().any()
               else 'n/a (centroid)')
        print(f'  {cond:>10} {lvl:>14}  n={len(g):>6,}  '
              f'ppm median {g["ppm"].median():+6.2f}  '
              f'resolution {res:>14}  '
              f'iso cos {g["isotope_cosine"].median():.3f}  '
              f'n_iso {g["n_isotopes_observed"].median():.0f}')
    print(f'\ndone -> {_out_dir()}')
    return table


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('data_root', nargs='?', default=None)
    ap.add_argument('--use-raw', action='store_true',
                    help='open the raw files (expensive); off by default')
    ap.add_argument('--force', action='store_true',
                    help='redo measurements already cached as parquet')
    ap.add_argument('--n-precursors', type=int, default=None)
    ap.add_argument('--top-intensity-frac', type=float, default=None,
                    help='sample only the top fraction of precursors by '
                         'abundance (default 0.30)')
    ap.add_argument('--extra', action='append', default=None, metavar='LABEL=PATH')
    ap.add_argument('--raw-dir', action='append', default=None,
                    help='additional directory to search for raw files')
    ap.add_argument('--cache-dir', default=None,
                    help='full-profile HDF caches and per-run measurement '
                         'tables, SHARED with the other raw-file scripts '
                         '(default: <data root>/%s)' % CACHE_SUBDIR)
    ap.add_argument('--sciex-conditions', default=None,
                    help='comma-separated conditions to predict with the 8600 '
                         'transfer-learned PeptDeep model instead of the '
                         'built-in (Orbitrap-trained) one')
    a = ap.parse_args()

    kw = {}
    if a.data_root:
        kw['data_root'] = a.data_root
    if a.n_precursors:
        kw['n_precursors'] = a.n_precursors
    if a.top_intensity_frac is not None:
        kw['top_intensity_frac'] = a.top_intensity_frac
    if a.use_raw:
        kw['use_raw'] = True
    if a.force:
        kw['force'] = True
    if a.cache_dir:
        kw['cache_dir'] = a.cache_dir
    if a.extra:
        kw['extra_conditions'] = dict(item.split('=', 1) for item in a.extra)
    if a.sciex_conditions:
        for c in a.sciex_conditions.split(','):
            PEPTDEEP_MODEL_BY_CONDITION[c.strip()] = PEPTDEEP_SCIEX_MODEL
    if a.raw_dir:
        EXTRA_RAW_DIRS.extend(a.raw_dir)
    configure(**kw)
    main()
