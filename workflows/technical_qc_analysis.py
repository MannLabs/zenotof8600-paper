"""Default technical QC analysis for multi-condition DIA experiments."""

import glob
import os
import re
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import zenotof8600_analytics as core


DATA_ROOT = os.path.join('Input_files', 'PEAKS')

CONDITIONS = None

SPLIT_RUNS = None

COND_ORDER = None

COND_LABELS = None

OUT_DIR = None

INCLUDE_ONLY = None
EXCLUDE = ('analysis', 'img', 'source_data', 'Mixed_species')

RAW_SEARCH_DIRS = ('.', '*', '..')
RAW_PATTERNS = ('*.wiff', '*.raw', '*.d')

ACQ_TIME_SOURCE = 'auto'

MIN_PLAUSIBLE_RUN_GAP_S = 60.0

PALETTE = core.PALETTE_SINGLE
COLOR = {}

SHOW_FIGURES = False

CV_THRESHOLD = 0.20

CV_ECDF_LEVEL = 'precursor'

LABEL_OFFSET = 0.045

PCA_HIGHLIGHT = []
HIGHLIGHT_COLOR = '#D62728'

OUTLIER_K = 3.0

MIN_REPS_FOR_FLAGGING = 4

MIN_RUNS_PER_LEVEL = 3

PC_FACTORS = [
    ('condition',           'condition2',        'categorical'),
    ('acq order (all)',     'acq_order',         'continuous'),
    ('acq order (in cond)', 'acq_order_within',  'continuous'),
    ('plate row',           'plate_row',         'categorical'),
    ('plate column',        'plate_col',         'continuous'),
]


_PEAKS_CANDIDATES = core.PEAKS_CANDIDATES
_DIANN_REJECT = core.DIANN_REJECT
_pick_output_file = core.pick_output_file


def discover_conditions(data_root=None, *, include_only=None, exclude=None):
    """{condition label: search-output path} from one subfolder per condition."""
    data_root = DATA_ROOT if data_root is None else data_root
    include_only = INCLUDE_ONLY if include_only is None else include_only
    exclude = EXCLUDE if exclude is None else exclude
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f'DATA_ROOT does not exist: {data_root}')

    found, skipped = {}, []
    for name in sorted(os.listdir(data_root)):
        folder = os.path.join(data_root, name)
        if not os.path.isdir(folder) or name in (exclude or ()):
            continue
        if include_only is not None and name not in include_only:
            continue
        path = _pick_output_file(folder)
        if path is None:
            skipped.append(name)
        else:
            found[name] = path
    if not found:
        raise FileNotFoundError(
            f'no search output found in any subfolder of {data_root}. Expected '
            f'one of {_PEAKS_CANDIDATES} or a DIA-NN combined report parquet.')
    if skipped:
        print(f'  skipped (no search output): {", ".join(skipped)}')
    return found


def condition_path(cond):
    """The search-output file of a condition, whatever form CONDITIONS holds."""
    value = CONDITIONS[cond]
    return value[0] if isinstance(value, (tuple, list)) else value


def condition_runs(cond):
    """The runs of a condition, or None when it is all of the file's runs."""
    value = CONDITIONS[cond]
    if not isinstance(value, (tuple, list)):
        return None
    runs = value[1]
    return None if runs is None else list(runs)


def _runs_of(path):
    from zenotof8600_analytics.io.experiments import _discover_runs
    return sorted(_discover_runs(path, core.detect_engine(path)))


def split_conditions(conditions, split_runs=None):
    """Expand {label: file} into per-run-group conditions."""
    split_runs = SPLIT_RUNS if split_runs is None else split_runs
    if not split_runs:
        return conditions

    from zenotof8600_analytics import plate_map

    single = len(conditions) == 1
    out, unmatched = {}, {}
    for label, value in conditions.items():
        path, only = ((value[0], value[1]) if isinstance(value, (tuple, list))
                      else (value, None))
        runs = _runs_of(path)
        if only is not None:
            runs = [r for r in runs if r in set(only)]

        groups, missed = {}, []
        for run in runs:
            if isinstance(split_runs, dict):
                keys = [k for k, pat in split_runs.items()
                        if re.search(pat, run)]
                if len(keys) > 1:
                    raise ValueError(
                        f'run {run!r} matches several split_runs patterns '
                        f'({keys}) — a run cannot be in two conditions')
                key = keys[0] if keys else None
            elif split_runs in plate_map.PLATE_FIELDS:
                key = plate_map.parse_run_position(run)[split_runs]
                key = None if key is None else str(key)
            else:
                m = re.search(split_runs, run)
                if m is not None and m.re.groups != 1:
                    raise ValueError(
                        f'split_runs regex {split_runs!r} must have exactly one '
                        f'capture group, has {m.re.groups}')
                key = m.group(1) if m else None
            if key is None:
                missed.append(run)
            else:
                groups.setdefault(key, []).append(run)

        if not groups:
            raise ValueError(
                f'{label}: split_runs {split_runs!r} matched none of the '
                f'{len(runs)} run names, e.g. {runs[0]!r}')
        for key in sorted(groups):
            new = key if single else f'{label}_{key}'
            out[new] = (path, groups[key])
        if missed:
            unmatched[label] = missed

    for label, missed in unmatched.items():
        print(f'  warning: {label}: {len(missed)} run(s) matched no group and '
              f'were dropped: {", ".join(missed[:4])}'
              + (' …' if len(missed) > 4 else ''))
    print(f'  split by {split_runs!r} -> {len(out)} conditions: '
          f'{", ".join(out)}')
    return out


def configure(*, data_root=None, conditions=None, cond_order=None,
              out_dir=None, include_only=None, exclude=None,
              outlier_k=None, pca_highlight=None, pc_factors=None,
              show_figures=None, split_runs=None, labels=None):
    """Set the module-level config and derive condition order + colours."""
    global DATA_ROOT, CONDITIONS, COND_ORDER, OUT_DIR, COLOR, COND_LABELS
    global INCLUDE_ONLY, EXCLUDE, OUTLIER_K, PCA_HIGHLIGHT, PC_FACTORS
    global SHOW_FIGURES, SPLIT_RUNS

    if show_figures is not None:
        SHOW_FIGURES = show_figures

    if data_root is not None:
        DATA_ROOT = data_root
    if include_only is not None:
        INCLUDE_ONLY = include_only
    if exclude is not None:
        EXCLUDE = exclude
    if outlier_k is not None:
        OUTLIER_K = outlier_k
    if pca_highlight is not None:
        PCA_HIGHLIGHT = list(pca_highlight)
    if pc_factors is not None:
        PC_FACTORS = list(pc_factors)
    if split_runs is not None:
        SPLIT_RUNS = split_runs
    if labels is not None:
        COND_LABELS = dict(labels)

    CONDITIONS = conditions if conditions is not None else (
        CONDITIONS if CONDITIONS is not None else discover_conditions())
    if SPLIT_RUNS and not any(isinstance(v, (tuple, list))
                              for v in CONDITIONS.values()):
        CONDITIONS = split_conditions(CONDITIONS)
        if cond_order is None and COND_ORDER is not None:
            COND_ORDER = None

    if COND_LABELS:
        unknown = [k for k in COND_LABELS
                   if k not in CONDITIONS and k not in COND_LABELS.values()]
        if unknown:
            raise ValueError(f'labels name unknown conditions: {unknown}')
        CONDITIONS = {COND_LABELS.get(k, k): v for k, v in CONDITIONS.items()}

    order = cond_order if cond_order is not None else (
        COND_ORDER if COND_ORDER is not None else list(CONDITIONS))
    if COND_LABELS:
        order = [COND_LABELS.get(c, c) for c in order]
    missing = [c for c in order if c not in CONDITIONS]
    if missing:
        raise ValueError(f'cond_order names unknown conditions: {missing}')
    COND_ORDER = order
    COLOR = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(COND_ORDER)}

    OUT_DIR = (out_dir if out_dir is not None else
               (OUT_DIR if OUT_DIR is not None
                else os.path.join(DATA_ROOT, 'analysis')))
    os.makedirs(OUT_DIR, exist_ok=True)
    if len(COND_ORDER) > len(PALETTE):
        print(f'  warning: {len(COND_ORDER)} conditions but only '
              f'{len(PALETTE)} palette hues — colours will repeat. Beyond ~6 '
              f'conditions this panel set needs rethinking anyway.')
    return CONDITIONS


@lru_cache(maxsize=None)
def sd_median_bias(n):
    """median(s)/sigma for a sample SD from `n` normal values."""
    if n is None or n < 2:
        return np.nan
    from scipy.stats import chi2
    return np.sqrt(chi2.median(n - 1) / (n - 1))


def n_corrected_cv(cv_pct, n_from, n_to=8):
    """Rescale a median CV measured over `n_from` replicates to `n_to`."""
    return cv_pct * sd_median_bias(n_to) / sd_median_bias(n_from)


def _lighten(color, factor=0.45):
    import matplotlib.colors as mcolors
    c = mcolors.to_rgb(color)
    return tuple(1 - (1 - ch) * (1 - factor) for ch in c)


def _robust_threshold(values, k=None, rel_floor=1.35, abs_floor=0.0):
    """Robust outlier cutoff: median + k * scaled MAD, with floors."""
    v = np.asarray(values, dtype=float)
    k = OUTLIER_K if k is None else k
    m = float(np.median(v))
    mad = float(np.median(np.abs(v - m))) * 1.4826
    return max(m + k * mad, m * rel_floor, abs_floor)


def _pca_outliers(pca_df, *, pcs=('PC1', 'PC2'), k=None, rel_floor=2.0):
    """Runs sitting far from their own condition's centroid in PC space."""
    cols = list(pcs)
    resid, eligible = [], []
    for _, sub in pca_df.groupby('condition2'):
        centre = sub[cols].median().to_numpy()
        d = pd.Series(np.linalg.norm(sub[cols].to_numpy() - centre, axis=1),
                      index=sub['run'].to_numpy())
        resid.append(d)
        if len(sub) >= MIN_REPS_FOR_FLAGGING:
            eligible.append(d)
    norms = pd.concat(resid)
    if not eligible:
        return norms, np.inf
    pooled = pd.concat(eligible).to_numpy()
    return norms, _robust_threshold(pooled, k=k, rel_floor=rel_floor)


def _raw_file_map(runs=None):
    """{run stem: raw file path} for raw files found near DATA_ROOT."""
    wanted = None if runs is None else set(runs)
    paths = {}
    for rel in RAW_SEARCH_DIRS:
        base = os.path.normpath(os.path.join(DATA_ROOT, rel))
        for pattern in RAW_PATTERNS:
            for p in glob.glob(os.path.join(base, pattern)):
                stem = os.path.basename(p)
                stem = stem[:stem.rindex('.')] if '.' in stem else stem
                if wanted is not None and stem not in wanted:
                    continue
                paths.setdefault(stem, p)
    return paths


def _add_acquisition_order(sample_info):
    """Attach acquisition order from raw-file modification times."""
    stamps, source = _acquisition_stamps(_raw_file_map(sample_info['run']))
    t = sample_info['run'].map(stamps)
    missing = int(t.isna().sum())
    if missing:
        print(f'  note: no raw file found for {missing}/{len(t)} run(s) in '
              f'{", ".join(RAW_SEARCH_DIRS)} — acquisition-order factors will '
              f'be skipped')
    elif source == 'mtime' and _copied_gap(t.dropna()) is not None:
        gap = _copied_gap(t.dropna())
        print(f'  note: raw-file mtimes are {gap / 60:.1f} min apart at the '
              f'median — too close together to be acquisitions, so these are '
              f'copy times. Acquisition-order factors will be skipped')
        t = pd.Series(np.nan, index=t.index)
    elif len(t):
        gaps = np.diff(np.sort(t.to_numpy())) / 60
        label = ('instrument timestamps' if source == 'instrument'
                 else 'raw-file mtimes')
        print(f'  acquisition order from {label} '
              f'(run-to-run gaps {gaps.min():.0f}-{gaps.max():.0f} min)')
    sample_info['acq_time'] = t
    sample_info['acq_order'] = t.rank(method='first')
    sample_info['acq_order_within'] = (t.groupby(sample_info['condition2'])
                                        .rank(method='first'))
    return sample_info


def _acquisition_stamps(paths):
    """({run: epoch seconds}, source label) for a {run: raw path} map."""
    if not paths:
        return {}, 'mtime'
    if ACQ_TIME_SOURCE in ('auto', 'instrument'):
        stamps = {}
        try:
            from zenotof8600_analytics.raw.metadata import acquisition_times
            for run, when in acquisition_times(list(paths.values())).items():
                if run in paths and pd.notna(when):
                    stamps[run] = pd.Timestamp(when).timestamp()
        except Exception as exc:
            print(f'  note: could not read instrument timestamps ({exc})')
        if len(stamps) == len(paths):
            return stamps, 'instrument'
        if ACQ_TIME_SOURCE == 'instrument':
            print(f'  note: instrument timestamps for only {len(stamps)} of '
                  f'{len(paths)} run(s) — acquisition-order factors will be '
                  f'skipped')
            return stamps, 'instrument'
    return {stem: os.path.getmtime(p) for stem, p in paths.items()}, 'mtime'


def _copied_gap(t):
    """Median run-to-run gap in seconds when it is too short to be real, else None."""
    v = np.sort(np.asarray(t, dtype=float))
    if v.size < 2:
        return None
    gap = float(np.median(np.diff(v)))
    return gap if gap < MIN_PLAUSIBLE_RUN_GAP_S else None


def _add_plate_position(sample_info):
    """Attach the plate position parsed out of each run name."""
    from zenotof8600_analytics import plate_map

    pos = plate_map.plate_metadata(sample_info['run'])
    for col in plate_map.PLATE_FIELDS:
        sample_info[col] = pos[col].to_numpy()
    n = int(sample_info['plate_well'].notna().sum())
    if n < len(sample_info):
        print(f'  note: no plate well parsed from {len(sample_info) - n}/'
              f'{len(sample_info)} run name(s) — plate factors will be skipped')
    return sample_info


def _eta_squared(y, groups):
    """Fraction of variance in `y` explained by a categorical factor."""
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return np.nan
    ss_between = 0.0
    for g in pd.unique(groups):
        v = y[groups == g]
        if v.size:
            ss_between += v.size * (v.mean() - y.mean()) ** 2
    return float(ss_between / ss_tot)


def _factor_r2(y, x, kind):
    """Variance in `y` explained by factor `x`, bias-corrected."""
    y = np.asarray(y, dtype=float)
    if kind == 'categorical':
        x = pd.Series(x).reset_index(drop=True)
        ok = x.notna().to_numpy() & np.isfinite(y)
        n, k = int(ok.sum()), int(x[ok].nunique())
        if k < 2 or n - k < 2:
            return np.nan
        return _adjusted_r2(_eta_squared(y[ok], x[ok].to_numpy()), n, k)
    x = pd.to_numeric(pd.Series(x), errors='coerce').to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.nanstd(x[ok]) == 0:
        return np.nan
    r2 = float(np.corrcoef(y[ok], x[ok])[0, 1] ** 2)
    return _adjusted_r2(r2, int(ok.sum()), 2)


def _adjusted_r2(r2, n, k):
    """R^2 corrected for the k parameters a factor spends on n observations."""
    if not np.isfinite(r2) or n - k < 2:
        return np.nan
    return float(max(0.0, 1 - (1 - r2) * (n - 1) / (n - k)))


def _scree_on_ax(ax, ve):
    """Variance spectrum: per-PC bars plus the cumulative curve."""
    pcs = np.arange(1, len(ve) + 1)
    cum = np.cumsum(ve)
    ax.bar(pcs, ve, color='#4d4d4d', edgecolor='black', linewidth=0.5,
           width=0.7, label='per PC')
    ax.plot(pcs, cum, color='black', lw=1.2, marker='o', ms=3.5,
            label='cumulative')
    for p, c in zip(pcs, cum):
        if p in (1, 2, len(ve)):
            ax.annotate(f'{c:.0f}%', (p, c), fontsize=6.5, xytext=(0, 4),
                        textcoords='offset points', ha='center')
    ax.set_xticks(pcs)
    ax.set_xlabel('Principal component', fontsize=10)
    ax.set_ylabel('Variance explained [%]', fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title('Variance spectrum\n(how many real dimensions)',
                 fontsize=11, fontweight='bold')
    ax.legend(frameon=False, fontsize=7.5, loc='center right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return [{'panel': 'variance_spectrum', 'pc': int(p),
             'variance_explained_pct': float(v),
             'cumulative_pct': float(c)}
            for p, v, c in zip(pcs, ve, cum)]


def _factor_heatmap_on_ax(ax, pca_df, ve):
    """Adjusted R^2 of each PC score against each metadata factor."""
    factors = [f for f in PC_FACTORS if f[1] in pca_df.columns
               and pca_df[f[1]].notna().any()]
    kept = []
    for label, col, kind in factors:
        if kind == 'categorical':
            n = int(pca_df[col].notna().sum())
            k = int(pca_df[col].nunique())
            if k < 2:
                print(f'  note: factor {label!r} skipped — only one level')
                continue
            if n / k < MIN_RUNS_PER_LEVEL:
                print(f'  note: factor {label!r} skipped — {k} levels over {n} '
                      f'runs is under {MIN_RUNS_PER_LEVEL} runs per level, so a '
                      f'{label} effect is not separable from run-to-run noise')
                continue
        kept.append((label, col, kind))
    factors = kept
    if not factors:
        ax.axis('off')
        ax.text(0.5, 0.5, 'no metadata factors available', ha='center',
                va='center', transform=ax.transAxes, fontsize=9,
                color='#555555')
        return []
    pcs = [c for c in pca_df.columns if c.startswith('PC') and c[2:].isdigit()]
    pcs = sorted(pcs, key=lambda c: int(c[2:]))
    mat = np.full((len(factors), len(pcs)), np.nan)
    rows = []
    for i, (label, col, kind) in enumerate(factors):
        for j, pc in enumerate(pcs):
            mat[i, j] = _factor_r2(pca_df[pc].to_numpy(), pca_df[col], kind)
            rows.append({'panel': 'pc_factor_r2', 'factor': label,
                         'pc': int(pc[2:]), 'r2': float(mat[i, j])})
    im = ax.imshow(mat, cmap='Reds', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(pcs)))
    ax.set_xticklabels([f'PC{p[2:]}\n{v:.0f}%' for p, v in zip(pcs, ve)],
                       fontsize=6.5)
    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels([f[0] for f in factors], fontsize=7.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center',
                        fontsize=6,
                        color='white' if mat[i, j] > 0.6 else '#333333')
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('adjusted R$^2$', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title('What explains each PC\n(adjusted R$^2$ vs metadata)',
                 fontsize=11, fontweight='bold')
    for s in ax.spines.values():
        s.set_visible(False)
    return rows


def _gene_label(pg, lookup, maxlen=14):
    """Short display label for a protein group: first gene symbol, plus a count of the remaining members."""
    genes = [g.strip() for g in str(lookup.get(pg, '') or '').split(';')
             if g.strip()]
    label = genes[0] if genes else str(pg).split(';')[0][:maxlen]
    if len(genes) > 1:
        label = f'{label} +{len(genes) - 1}'
    return label[:maxlen + 4]


def _pc1_loadings_on_ax(ax, res, df, pca_df, *, top_n=12):
    """Protein groups with the largest |PC1 loading|, signed."""
    pc1 = res['loadings']['PC1'].dropna()
    top = pc1.reindex(pc1.abs().sort_values(ascending=False).index[:top_n])
    top = top.sort_values()
    lookup = (df.drop_duplicates('protein_group')
                .set_index('protein_group')['genes'])
    labels = [_gene_label(pg, lookup) for pg in top.index]
    y = np.arange(len(top))
    ax.barh(y, top.to_numpy(), height=0.7, color='#4d4d4d',
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color='black', lw=0.6)
    ax.set_ylim(-0.8, len(top) - 0.2 + 3.6)

    def _short(name, n=18):
        return name if len(name) <= n else name[:n - 1] + '…'

    med_pc1 = pca_df.groupby('condition2')['PC1'].median()
    med_pc1 = med_pc1.reindex([c for c in COND_ORDER if c in med_pc1.index])
    neg = list(med_pc1[med_pc1 < 0].index) or [med_pc1.idxmin()]
    pos = list(med_pc1[med_pc1 >= 0].index) or [med_pc1.idxmax()]

    def _side_label(conds, n_max=3):
        shown = [_short(c, 16) for c in conds[:n_max]]
        extra = len(conds) - len(shown)
        return ', '.join(shown) + (f' +{extra}' if extra else '')

    if len(med_pc1) < 2:
        ax.text(0.5, 0.99, 'single condition — PC1 separates replicate\n'
                'runs, so its sign is not a contrast',
                transform=ax.transAxes, ha='center', va='top', fontsize=7,
                color='#777777')
    else:
        ax.text(0.02, 0.99, f'$\\leftarrow$ higher in {_side_label(neg)}',
                transform=ax.transAxes, ha='left', va='top', fontsize=7,
                color=COLOR.get(neg[0], 'black') if len(neg) == 1 else 'black')
        ax.text(0.98, 0.945, f'higher in {_side_label(pos)} $\\rightarrow$',
                transform=ax.transAxes, ha='right', va='top', fontsize=7,
                color=COLOR.get(pos[0], 'black') if len(pos) == 1 else 'black')

    share = float((top.to_numpy() ** 2).sum())
    uniform = len(top) / len(pc1)
    ax.text(0.5, 0.885,
            f'top {len(top)} carry {share:.1%} of PC1 '
            f'(uniform share: {uniform:.1%})',
            transform=ax.transAxes, ha='center', va='top', fontsize=6.5,
            color='#555555')

    ax.set_xlabel('PC1 loading', fontsize=10)
    ax.set_title(f'Drivers of PC1\n(top {len(top)} protein groups)',
                 fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return [{'panel': 'pc1_loadings', 'rank': int(i + 1),
             'protein_group': str(pg), 'gene_label': lab,
             'pc1_loading': float(v),
             'pc2_loading': float(res['loadings']['PC2'].get(pg, np.nan))}
            for i, (pg, v, lab) in enumerate(
                zip(top.index[::-1], top.to_numpy()[::-1], labels[::-1]))]


def _matches_highlight(run, patterns):
    tail = run.rsplit('_', 1)[-1]
    return any(p == run or p == tail for p in patterns)


def _compact_legend(ax, conditions, *, loc='best', fontsize=7.5, labels=None,
                    title=None, title_fontsize=6.5, alpha=None,
                    bbox_to_anchor=None, ncol=1):
    """House key: black label text with a small colour swatch, frameless and tightly padded so it costs little panel space."""
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLOR[c], edgecolor='black', linewidth=0.4,
                     alpha=alpha)
               for c in conditions]
    leg = ax.legend(handles, list(labels if labels is not None else conditions),
                    loc=loc, frameon=False, title=title, fontsize=fontsize,
                    handlelength=0.9, handleheight=0.9, handletextpad=0.4,
                    borderpad=0.2, labelspacing=0.35,
                    bbox_to_anchor=bbox_to_anchor, ncol=ncol)
    if title is not None:
        leg.get_title().set_fontsize(title_fontsize)
        leg.get_title().set_color('#666666')
    return leg


def _condition_xticklabels(ax, order, *, fontsize=9):
    """Condition names on a categorical x-axis, rotated only when they must be."""
    ax.set_xticks(range(len(order)))
    axes_in = ax.figure.get_size_inches()[0] * ax.get_position().width
    slot_in = axes_in / max(len(order), 1)
    widest_in = max((len(str(c)) for c in order), default=0) * 0.6 * fontsize / 72
    crowded = widest_in > 0.9 * slot_in
    ax.set_xticklabels(order, fontsize=fontsize,
                       **(dict(rotation=20, ha='right') if crowded else {}))
    half = 0.5 if len(order) > 1 else 1.0
    ax.set_xlim(-half, len(order) - 1 + half)


def _unbold_axis_labels(ax):
    """House style: only titles are bold."""
    for lbl in (ax.xaxis.label, ax.yaxis.label):
        lbl.set_fontweight('normal')
        lbl.set_fontsize(10)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight('normal')


def save(fig, name, source_df=None):
    """Save a figure as PDF (+PNG for quick viewing) and its source data."""
    pdf = os.path.join(OUT_DIR, f'{name}.pdf')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, f'{name}.png'), dpi=300,
                bbox_inches='tight')
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)
    if source_df is not None:
        source_df.to_csv(os.path.join(OUT_DIR, f'{name}_sourcedata.csv'),
                         index=False)
    print(f'  wrote {name}.pdf' + ('  + sourcedata' if source_df is not None else ''))


def build_experiments():
    """One experiment dict per run, engine-agnostic."""
    experiment_name = os.path.basename(os.path.normpath(DATA_ROOT)) or 'experiment'
    exps = []
    for cond in COND_ORDER:
        path = condition_path(cond)
        engine = core.detect_engine(path)
        runs = condition_runs(cond) or _runs_of(path)
        if not runs:
            raise ValueError(f'{cond}: no runs found in {path}')
        folder = os.path.basename(os.path.dirname(path)) or path
        print(f'  {cond}: {len(runs)} runs ({engine}, {folder}/'
              f'{os.path.basename(path)})')
        for i, r in enumerate(runs):
            exps.append({
                'path': path,
                'file_tags': r,
                'condition1': experiment_name,
                'condition2': cond,
                'replicate': i + 1,
            })
    return exps


def _disambiguate_runs(df, sample_info):
    """Make `run` unique across conditions, tagging it where it is not."""
    per_run = sample_info.groupby('run')['condition2'].nunique()
    clashing = set(per_run[per_run > 1].index)
    if not clashing:
        return df, sample_info

    key = (sample_info[sample_info['run'].isin(clashing)]
           .groupby(['run', 'engine'])['condition2'].agg(['first', 'nunique']))
    ambiguous = key[key['nunique'] > 1]
    if len(ambiguous):
        raise ValueError(
            'these (run, engine) pairs appear in more than one condition, so '
            'the conditions cannot be separated in `df`:\n'
            f'{ambiguous.index.tolist()}')

    rename = {k: f'{k[0]} [{c}]' for k, c in key['first'].items()}
    for frame in (df, sample_info):
        pairs = list(zip(frame['run'], frame['engine']))
        frame['run'] = [rename.get(p, p[0]) for p in pairs]
    print(f'  note: {len(clashing)} run name(s) shared by several conditions — '
          f'tagged with the condition to keep them apart')
    return df, sample_info


def load(**configure_kwargs):
    """Configure, then load every condition into the canonical long DataFrame."""
    configure(**configure_kwargs)
    core.init_plotting()
    core.set_default_panel_size(4, 4)
    print(f'loading from {DATA_ROOT} ...')
    df, sample_info = core.load_experiments(build_experiments())
    df, sample_info = _disambiguate_runs(df, sample_info)
    sample_info = _add_acquisition_order(sample_info)
    sample_info = _add_plate_position(sample_info)
    print(f'  long df: {df.shape[0]:,} rows x {df.shape[1]} cols, '
          f'{len(sample_info)} runs, {len(COND_ORDER)} conditions')
    return df, sample_info


def summary_tables(df, sample_info):
    per_run = core.process_experiment(df, sample_info, group_col='condition2',
                                      min_values_for_cv=3)
    keep = ['run', 'condition2', 'replicate', 'precursor', 'peptide',
            'protein_group', 'total_intensity', 'log2_total_intensity',
            'avg_MC', 'Pr20', 'PG20',
            'total_precursors', 'total_peptides', 'total_protein_groups']
    keep += [c for c in per_run.columns if c.startswith('MC')]
    per_run = per_run[[c for c in keep if c in per_run.columns]]
    per_run = per_run.sort_values(['condition2', 'run'])

    agg = (per_run.groupby('condition2')
           .agg(n_runs=('run', 'nunique'),
                precursors_mean=('precursor', 'mean'),
                precursors_sd=('precursor', 'std'),
                peptides_mean=('peptide', 'mean'),
                peptides_sd=('peptide', 'std'),
                protein_groups_mean=('protein_group', 'mean'),
                protein_groups_sd=('protein_group', 'std'),
                precursors_union=('total_precursors', 'first'),
                peptides_union=('total_peptides', 'first'),
                protein_groups_union=('total_protein_groups', 'first'),
                precursors_cv20=('Pr20', 'first'),
                protein_groups_cv20=('PG20', 'first'),
                avg_missed_cleavages=('avg_MC', 'mean'))
           .reindex(COND_ORDER).reset_index())
    return per_run, agg


CV_SCREEN_LEVELS = [
    dict(key='precursor', label='Precursors', y_label='# precursors',
         id_col='precursor_id', int_col='precursor_intensity',
         src_col='precursors', ecdf_series='precursor', short='precursor',
         mean_col='mean_precursors_per_run',
         union_col='union_precursors', cv20_col='precursors_cv20',
         median_col='median_precursor_cv_pct',
         median_n8_col='median_precursor_cv_pct_n8',
         frac_col='frac_precursor_cv_below_20'),
    dict(key='peptide', label='Peptides', y_label='# peptides',
         id_col='peptide_id', int_col='peptide_intensity',
         src_col='peptides', ecdf_series='peptide', short='peptide',
         mean_col='mean_peptides_per_run',
         union_col='union_peptides', cv20_col='peptides_cv20',
         median_col='median_peptide_cv_pct',
         median_n8_col='median_peptide_cv_pct_n8',
         frac_col='frac_peptide_cv_below_20'),
    dict(key='protein', label='Protein groups', y_label='# protein groups',
         id_col='protein_group', int_col='pg_intensity',
         src_col='protein_groups', ecdf_series='protein_group', short='PG',
         mean_col='mean_protein_groups_per_run',
         union_col='union_protein_groups', cv20_col='protein_groups_cv20',
         median_col='median_pg_cv_pct', median_n8_col='median_pg_cv_pct_n8',
         frac_col='frac_pg_cv_below_20'),
]


def _ecdf_first():
    """`CV_SCREEN_LEVELS` with the ECDF's own level moved to the front."""
    return (sorted(CV_SCREEN_LEVELS, key=lambda l: l['key'] != CV_ECDF_LEVEL))


def _overlapping_bars_on_ax(ax, order, totals_by_cond, union_by_cond,
                            cv20_by_cond, *, y_label, title,
                            bar_width=0.6, lighten_factor=0.45):
    """Light bar = union of the condition's runs, dark bar = its CV<20% subset."""
    x = np.arange(len(order))
    max_y = max(max(union_by_cond[c], max(totals_by_cond[c])) for c in order)
    rng = np.random.default_rng(0)
    inner = []
    for j, cond in enumerate(order):
        color = COLOR[cond]
        totals = np.asarray(totals_by_cond[cond], dtype=float)
        higher, lower = float(union_by_cond[cond]), float(cv20_by_cond[cond])
        ax.bar(x[j], higher, width=bar_width, color=_lighten(color, lighten_factor),
               edgecolor='darkgray', linewidth=1.0)
        ax.bar(x[j], lower, width=bar_width, color=color,
               edgecolor='black', linewidth=1.0)
        jitter = rng.uniform(-bar_width * 0.18, bar_width * 0.18, totals.size)
        ax.scatter(x[j] + jitter, totals, c='black', s=22, alpha=0.75,
                   zorder=5, edgecolors='white', linewidths=0.5)
        ax.text(x[j], higher + max_y * LABEL_OFFSET, f'{int(round(higher)):,}',
                ha='center', va='bottom', fontsize=8)
        inner.append((ax.text(x[j], lower - max_y * 0.012,
                              f'{int(lower):,}\nCV<20%', ha='center', va='top',
                              fontsize=8, color='white'), lower))
    _condition_xticklabels(ax, order, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylim(0, max_y * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return inner


def _fit_inner_bar_labels(fig, entries, *, bar_width=0.6):
    """Rotate a bar's inner label upright when it is wider than the bar."""
    fig.canvas.draw()
    rotated = []
    for ax, texts in entries:
        px = abs(ax.transData.transform((bar_width, 0))[0]
                 - ax.transData.transform((0, 0))[0])
        for t, top in texts:
            if t.get_window_extent().width <= px * 0.95:
                continue
            t.set_text(' '.join(t.get_text().split('\n')))
            t.set_rotation(90)
            rotated.append((ax, t, top))
    if not rotated:
        return
    fig.canvas.draw()
    for ax, t, top in rotated:
        bar_px = abs(ax.transData.transform((0, top))[1]
                     - ax.transData.transform((0, 0))[1])
        if t.get_window_extent().height <= bar_px * 0.95:
            continue
        t.set_y(top + ax.get_ylim()[1] * 0.012)
        t.set_va('bottom')
        t.set_color('black')


PANEL_NAME = 'ids_and_cv_by_level'


def cv_screen(df, sample_info):
    """The house CV screen: one bar panel per level, plus a CV ECDF."""
    quant = df[df['precursor_intensity'].notna() & (df['precursor_intensity'] > 0)]

    run_names = {c: list(sample_info.loc[sample_info['condition2'] == c, 'run'])
                 for c in COND_ORDER}

    totals, union, cv20, cvs, med = {}, {}, {}, {}, {}
    for lvl in CV_SCREEN_LEVELS:
        key = lvl['key']
        cv_tab = core._compute_cv_table(df, sample_info, level=key,
                                        group_col='condition2',
                                        min_values_for_cv=3)
        sub = quant[quant[lvl['int_col']].notna() & (quant[lvl['int_col']] > 0)]
        for cond in COND_ORDER:
            totals[(key, cond)] = [
                int(sub.loc[sub['run'] == r, lvl['id_col']].nunique())
                for r in run_names[cond]]
            union[(key, cond)] = int(
                sub.loc[sub['run'].isin(run_names[cond]), lvl['id_col']].nunique())
            v = cv_tab.loc[cv_tab['group'] == cond, 'cv'].values
            cvs[(key, cond)] = v
            cv20[(key, cond)] = int((v < CV_THRESHOLD).sum())
            med[(key, cond)] = float(np.median(v) * 100) if v.size else np.nan

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.ravel()
    bar_labels = []
    for ax, lvl in zip(axes, CV_SCREEN_LEVELS):
        key = lvl['key']
        bar_labels.append((ax, _overlapping_bars_on_ax(
            ax, COND_ORDER,
            {c: totals[(key, c)] for c in COND_ORDER},
            {c: union[(key, c)] for c in COND_ORDER},
            {c: cv20[(key, c)] for c in COND_ORDER},
            y_label=lvl['y_label'],
            title=f"{lvl['label']}\n(union of runs vs CV<20%)")))

    thr = CV_THRESHOLD * 100
    ax = axes[len(CV_SCREEN_LEVELS)]
    for cond in COND_ORDER:
        v = cvs[(CV_ECDF_LEVEL, cond)]
        if v.size == 0:
            continue
        xs = np.sort(v) * 100
        ys = np.arange(1, xs.size + 1) / xs.size
        meds = ' / '.join(f'{med[(l["key"], cond)]:.1f}' for l in _ecdf_first())
        ax.plot(xs, ys, color=COLOR[cond], ls='-', lw=1.5,
                label=f'{cond} ({meds}%)')
    guide = ax.axvline(thr, color='#d62728', ls=':', lw=1.2)
    below = [float((cvs[(CV_ECDF_LEVEL, c)] < CV_THRESHOLD).mean())
             for c in COND_ORDER if cvs[(CV_ECDF_LEVEL, c)].size]
    ax.text(thr + 0.8, max(min(below) - 0.03, 0.06) if below else 0.5,
            f'CV = {thr:g}%', color='#d62728', fontsize=8, ha='left', va='top')
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.01)
    ax.set_xticks(np.arange(0, 61, 5))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_yticks(np.arange(0, 1.01, 0.05), minor=True)
    ax.set_xlabel('CV [%]', fontsize=10)
    ax.set_ylabel('Cumulative fraction', fontsize=10)
    ax.set_title(f'ECDF of {CV_ECDF_LEVEL} CVs\n(median CV per level in legend)',
                 fontsize=11, fontweight='bold')
    leg = ax.legend(frameon=False, fontsize=8, loc='lower right',
                    title='median CV: ' + ' / '.join(l['short']
                                                     for l in _ecdf_first()),
                    title_fontsize=8, alignment='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(ax.transAxes.inverted())
    lo, hi = ax.get_xlim()
    if bb.x0 <= (thr - lo) / (hi - lo) <= bb.x1:
        guide.set_ydata([min(bb.y1 + 0.02, 1.0), 1.0])

    _fit_inner_bar_labels(fig, bar_labels)

    rows = []
    for cond in COND_ORDER:
        for i, r in enumerate(run_names[cond]):
            rows.append({'panel': 'bars', 'condition': cond, 'run': r,
                         'series': 'per-run total',
                         **{l['src_col']: totals[(l['key'], cond)][i]
                            for l in CV_SCREEN_LEVELS}})
        rows.append({'panel': 'bars', 'condition': cond, 'run': 'UNION',
                     'series': 'bar height (light)',
                     **{l['src_col']: union[(l['key'], cond)]
                        for l in CV_SCREEN_LEVELS}})
        rows.append({'panel': 'bars', 'condition': cond, 'run': 'MEAN',
                     'series': 'per-run mean (not drawn)',
                     **{l['src_col']: float(np.mean(totals[(l['key'], cond)]))
                        for l in CV_SCREEN_LEVELS}})
        rows.append({'panel': 'bars', 'condition': cond, 'run': 'CV<20%',
                     'series': 'bar height (dark)',
                     **{l['src_col']: cv20[(l['key'], cond)]
                        for l in CV_SCREEN_LEVELS}})
        rows.append({'panel': 'bars', 'condition': cond, 'run': 'MEDIAN CV %',
                     'series': 'inline label',
                     **{l['src_col']: med[(l['key'], cond)]
                        for l in CV_SCREEN_LEVELS}})

    ecdf_rows = []
    for cond in COND_ORDER:
        for lvl in CV_SCREEN_LEVELS:
            xs = np.sort(cvs[(lvl['key'], cond)]) * 100
            ecdf_rows.append(pd.DataFrame(
                {'panel': 'ecdf', 'condition': cond,
                 'series': lvl['ecdf_series'],
                 'drawn': lvl['key'] == CV_ECDF_LEVEL,
                 'cv_percent': xs,
                 'cumulative_fraction': np.arange(1, xs.size + 1) / xs.size}))
    src = pd.concat([pd.DataFrame(rows)] + ecdf_rows, ignore_index=True)
    save(fig, PANEL_NAME, src)

    cv_summary = pd.DataFrame([{
        'condition2': cond,
        'n_replicates': len(run_names[cond]),
        **{l['mean_col']: float(np.mean(totals[(l['key'], cond)]))
           for l in CV_SCREEN_LEVELS},
        **{l['union_col']: union[(l['key'], cond)] for l in CV_SCREEN_LEVELS},
        **{l['cv20_col']: cv20[(l['key'], cond)] for l in CV_SCREEN_LEVELS},
        **{l['median_col']: med[(l['key'], cond)] for l in CV_SCREEN_LEVELS},
        **{l['median_n8_col']: n_corrected_cv(med[(l['key'], cond)],
                                              len(run_names[cond]))
           for l in CV_SCREEN_LEVELS},
        **{l['frac_col']: float((cvs[(l['key'], cond)] < CV_THRESHOLD).mean())
           for l in CV_SCREEN_LEVELS},
    } for cond in COND_ORDER])
    return cv_summary


def _detection_frequency(df, sample_info):
    """Per condition and level, the histogram of 'detected in k of N runs'."""
    quant = df[df['precursor_intensity'].notna() & (df['precursor_intensity'] > 0)]
    pg_rows = df[df['pg_intensity'].notna() & (df['pg_intensity'] > 0)]
    hist, n_reps = {}, {}
    for cond in COND_ORDER:
        runs = set(sample_info.loc[sample_info['condition2'] == cond, 'run'])
        n = len(runs)
        n_reps[cond] = n
        for level, src, col in (('precursor', quant, 'precursor_id'),
                                ('protein_group', pg_rows, 'protein_group')):
            sub = src[src['run'].isin(runs)]
            per_ent = sub.groupby(col)['run'].nunique()
            counts = per_ent.value_counts().reindex(range(1, n + 1), fill_value=0)
            hist[(cond, level)] = counts.to_numpy(dtype=float)
    return hist, n_reps


def completeness(df, sample_info):
    """Two-panel completeness figure."""
    hist, n_reps = _detection_frequency(df, sample_info)
    n = max(n_reps.values())
    equal_n = len(set(n_reps.values())) == 1
    xlabel = (f'Detected in at least k of {n} replicates' if equal_n
              else 'Detected in at least k of N replicates')

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax_pr = axes[0]
    ax_pg = ax_pr.twinx()
    rows = []
    for cond in COND_ORDER:
        k = np.arange(1, n_reps[cond] + 1)
        for level, ls in (('protein_group', '-'), ('precursor', '--')):
            h = hist[(cond, level)]
            total = h.sum()
            at_least = h[::-1].cumsum()[::-1]
            retained = at_least / total
            at_most = h.cumsum() / total
            label = (f'{cond} (n={n_reps[cond]})'
                     if level == 'protein_group' else '_nolegend_')
            target = ax_pg if level == 'protein_group' else ax_pr
            target.plot(k, at_least, color=COLOR[cond], ls=ls, lw=1.5,
                        marker='o', ms=3.5, label=label)
            axes[1].plot(k, retained, color=COLOR[cond], ls=ls, lw=1.5,
                         marker='o', ms=3.5, label=label)
            for ki, a, r, m, hi in zip(k, at_least, retained, at_most, h):
                rows.append({'condition': cond, 'level': level,
                             'k_replicates': int(ki),
                             'n_detected_in_exactly_k': int(hi),
                             'n_detected_in_at_least_k': int(a),
                             'fraction_retained_at_least_k': float(r),
                             'cumulative_fraction_at_most_k': float(m)})

    fmt = plt.FuncFormatter(lambda v, _: f'{int(v):,}')
    ax_pr.set_xlabel(xlabel, fontsize=10)
    ax_pr.set_ylabel('# precursors (dashed)', fontsize=10)
    ax_pg.set_ylabel('# protein groups (solid)', fontsize=10)
    ax_pr.set_title('Completeness filter\n(absolute IDs retained)',
                    fontsize=11, fontweight='bold')
    ax_pr.set_xticks(np.arange(1, n + 1))
    for a in (ax_pr, ax_pg):
        a.set_ylim(0, None)
        a.yaxis.set_major_formatter(fmt)
        a.spines['top'].set_visible(False)
    ax_pg.spines['right'].set_visible(True)
    handles = ax_pr.get_legend_handles_labels()[0] + ax_pg.get_legend_handles_labels()[0]
    labels = ax_pr.get_legend_handles_labels()[1] + ax_pg.get_legend_handles_labels()[1]
    ax_pr.legend(handles, labels, frameon=False, fontsize=8, loc='lower left')

    ax = axes[1]
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Fraction of union retained', fontsize=10)
    ax.set_title('Completeness curve\n(solid = PG, dashed = precursor)',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(np.arange(1, n + 1))
    ax.set_ylim(0, 1.01)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_yticks(np.arange(0, 1.01, 0.05), minor=True)
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    comp = pd.DataFrame(rows)
    save(fig, 'data_completeness', comp)
    return comp


N_BINS = 10

MIN_COMPLETE_CASES = 50


def _wide_complete(df, sample_info, cond, id_col, val_col):
    """Entities x runs matrix for one condition, complete cases only."""
    runs = sorted(sample_info.loc[sample_info['condition2'] == cond, 'run'])
    sub = df[df['run'].isin(runs)].drop_duplicates(['run', id_col])
    wide = (sub.pivot(index=id_col, columns='run', values=val_col)
               .reindex(columns=runs))
    return wide.dropna()


def replicate_variability(df, sample_info):
    """Two panels on replicate scatter, complete cases only."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    rows = []
    summary = []

    wides = {cond: _wide_complete(df, sample_info, cond, 'peptide_id',
                                  'peptide_intensity')
             for cond in COND_ORDER}
    drawn = []
    for cond in COND_ORDER:
        n_complete = len(wides[cond])
        if n_complete >= MIN_COMPLETE_CASES:
            drawn.append(cond)
            continue
        n_reps = int((sample_info['condition2'] == cond).sum())
        print(f'  note: {cond}: {n_complete} peptide(s) quantified in all '
              f'{n_reps} replicates (need >= {MIN_COMPLETE_CASES}) — skipped in '
              f'the replicate-variability panels')
        summary.append({'condition': cond, 'level': 'peptide',
                        'n_complete': n_complete,
                        'median_cv_pct': np.nan,
                        'top_decile_median_cv_pct': np.nan,
                        'bottom_decile_median_cv_pct': np.nan})

    ax = axes[0]
    for cond in drawn:
        wide = wides[cond]
        mean = wide.mean(axis=1)
        cv = wide.std(axis=1, ddof=1) / mean
        binned = pd.qcut(mean, N_BINS, labels=False, duplicates='drop')
        stats = (pd.DataFrame({'cv': cv, 'mean': mean, 'bin': binned})
                   .groupby('bin', observed=True)
                   .agg(median_cv=('cv', 'median'),
                        q25=('cv', lambda s: s.quantile(0.25)),
                        q75=('cv', lambda s: s.quantile(0.75)),
                        median_intensity=('mean', 'median'),
                        n=('cv', 'size')))
        x = np.log10(stats['median_intensity'])
        ax.plot(x, stats['median_cv'] * 100, color=COLOR[cond], lw=1.5,
                marker='o', ms=3.5, label=cond)
        ax.fill_between(x, stats['q25'] * 100, stats['q75'] * 100,
                        color=COLOR[cond], alpha=0.18, lw=0)
        for b, r in stats.iterrows():
            rows.append({'panel': 'cv_vs_abundance', 'condition': cond,
                         'level': 'peptide', 'intensity_decile': int(b) + 1,
                         'median_log10_intensity': float(np.log10(r['median_intensity'])),
                         'median_cv_pct': float(r['median_cv'] * 100),
                         'q25_cv_pct': float(r['q25'] * 100),
                         'q75_cv_pct': float(r['q75'] * 100),
                         'n_peptides': int(r['n'])})
        summary.append({'condition': cond, 'level': 'peptide',
                        'n_complete': int(len(cv)),
                        'median_cv_pct': float(cv.median() * 100),
                        'top_decile_median_cv_pct': float(stats['median_cv'].iloc[-1] * 100),
                        'bottom_decile_median_cv_pct': float(stats['median_cv'].iloc[0] * 100)})
    ax.axhline(20, color='#d62728', ls=':', lw=1.2)
    ax.text(ax.get_xlim()[0], 21, 'CV = 20%', color='#d62728', fontsize=8)
    ax.set_xlabel('log₁₀ mean peptide intensity', fontsize=10)
    ax.set_ylabel('Peptide CV [%]', fontsize=10)
    ax.set_title('CV vs abundance\n(median, band = IQR)',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8, loc='upper right')

    ax = axes[1]
    n_runs_total = int(sample_info['condition2'].isin(COND_ORDER).sum())
    msize = 34 if n_runs_total <= 24 else 14
    flagged_runs = []
    for cond in drawn:
        wide = wides[cond]
        dev = np.log2(wide.div(wide.median(axis=1), axis=0))
        med = dev.median()
        iqr = dev.quantile(0.75) - dev.quantile(0.25)
        if len(wide.columns) >= MIN_REPS_FOR_FLAGGING:
            thr_iqr = _robust_threshold(iqr.to_numpy())
            thr_bias = _robust_threshold(med.abs().to_numpy(), abs_floor=0.02)
        else:
            thr_iqr = thr_bias = np.inf
        ax.scatter(med.to_numpy(), iqr.to_numpy(), s=msize, color=COLOR[cond],
                   alpha=0.85, edgecolors='black', linewidths=0.5, zorder=4,
                   label=cond)
        if np.isfinite(thr_iqr):
            ax.axhline(thr_iqr, color=COLOR[cond], ls='--', lw=0.8, alpha=0.45,
                       zorder=1)
        for run in wide.columns:
            short = run.split('_')[-1]
            flag = bool(iqr[run] > thr_iqr or abs(med[run]) > thr_bias)
            if flag:
                flagged_runs.append((cond, short, float(med[run]),
                                     float(iqr[run])))
                ax.annotate(short, (med[run], iqr[run]), fontsize=7,
                            xytext=(4, 3), textcoords='offset points',
                            color=COLOR[cond], zorder=6)
            rows.append({'panel': 'per_run_bias_vs_scatter', 'condition': cond,
                         'level': 'peptide', 'run': run,
                         'median_log2_ratio': float(med[run]),
                         'iqr_log2_ratio': float(iqr[run]),
                         'sd_log2_ratio': float(dev[run].std(ddof=1)),
                         'iqr_threshold': float(thr_iqr),
                         'bias_threshold': float(thr_bias),
                         'flagged': flag,
                         'n_peptides': int(dev.shape[0])})
    ax.axvline(0, color='black', lw=0.6, alpha=0.6, zorder=1)
    if any(len(wides[c].columns) == 3 for c in drawn):
        ax.text(0.5, 0.03, 'bias pinned to 0 at n = 3 (each run is the median\n'
                'for a third of the entities) — read the y axis',
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=8, color='#777777')
        print('  note: bias axis is degenerate at 3 replicates '
              '(every run sits at exactly 0) — read the scatter axis')
    ax.set_xlabel('median log₂ (run / replicate median)', fontsize=10)
    ax.set_ylabel('IQR of log₂ (run / replicate median)', fontsize=10)
    ax.set_title('Per-run bias vs scatter\n(dashed = outlier threshold)',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, None)
    _compact_legend(ax, drawn, loc='best')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if not drawn:
            ax.text(0.5, 0.5, 'no condition has enough complete cases',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='#555555')
    fig.tight_layout()
    save(fig, 'replicate_variability', pd.DataFrame(rows))

    if flagged_runs:
        print('  flagged runs (bias or scatter beyond threshold):')
        for cond, short, m, i in flagged_runs:
            print(f'    {cond:10s} {short:5s} median {m:+.3f}  IQR {i:.3f}')
    else:
        print('  no runs flagged')
    return pd.DataFrame(summary)


_LEVEL_SPEC = {'precursor':     ('precursor_id', 'precursor_intensity', 'precursors'),
               'protein_group': ('protein_group', 'pg_intensity', 'protein groups')}


def _mean_intensity_per_entity(df, sample_info, cond, level):
    """Mean intensity of each entity across one condition's runs (skips NaN)."""
    id_col, val_col, _ = _LEVEL_SPEC[level]
    runs = set(sample_info.loc[sample_info['condition2'] == cond, 'run'])
    sub = df[df['run'].isin(runs)].drop_duplicates(['run', id_col])
    s = sub.groupby(id_col)[val_col].mean().dropna()
    return s[s > 0]


def _rank_on_ax(ax, df, sample_info, level, *, point_size=3, alpha=0.35):
    id_col, _, plural = _LEVEL_SPEC[level]
    rows = []
    for cond in COND_ORDER:
        s = _mean_intensity_per_entity(df, sample_info, cond, level)
        s = s.sort_values(ascending=False)
        y = np.log10(s.to_numpy())
        x = np.arange(1, y.size + 1)
        ax.scatter(x, y, s=point_size, alpha=alpha, color=COLOR[cond],
                   linewidths=0, label=f'{cond} (n={y.size:,})')
        step = max(1, y.size // 2000)
        for xi, yi in zip(x[::step], y[::step]):
            rows.append({'panel': f'rank_{level}', 'condition': cond,
                         'rank': int(xi), 'log10_mean_intensity': float(yi)})
    ax.set_xlabel(f'{plural.capitalize()[:-1]} rank (most $\\to$ least abundant)',
                  fontsize=10)
    ax.set_ylabel(f'log₁₀ mean {plural[:-1]} intensity', fontsize=10)
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.set_title(f'{plural.capitalize()}\n(rank / dynamic range)',
                 fontsize=11, fontweight='bold')
    leg = ax.legend(frameon=False, fontsize=8, loc='upper right',
                    markerscale=4)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return rows


DR_PERCENTILES = (1, 99)


def _intensity_hist_on_ax(ax, df, sample_info, level, *, bins=50, alpha=0.6):
    _, _, plural = _LEVEL_SPEC[level]
    series = {cond: np.log10(_mean_intensity_per_entity(df, sample_info, cond,
                                                        level).to_numpy())
              for cond in COND_ORDER}
    lo = min(v.min() for v in series.values())
    hi = max(v.max() for v in series.values())
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    rows, stats = [], {}
    for cond in COND_ORDER:
        v = series[cond]
        p_lo, p_hi = np.percentile(v, DR_PERCENTILES)
        span, full = p_hi - p_lo, v.max() - v.min()
        stats[cond] = (v.size, span, full)
        ax.hist(v, bins=edges, color=COLOR[cond], alpha=alpha,
                edgecolor='black', linewidth=0.5)
        counts, _ = np.histogram(v, bins=edges)
        for c, n_ in zip(centers, counts):
            rows.append({'panel': f'intensity_histogram_{level}',
                         'condition': cond, 'series': 'histogram',
                         'log10_mean_intensity_bin_center': float(c),
                         'count': int(n_)})
        rows.append({'panel': f'intensity_histogram_{level}',
                     'condition': cond, 'series': 'dynamic_range',
                     'n_entities': int(v.size),
                     f'p{DR_PERCENTILES[0]}_log10': float(p_lo),
                     f'p{DR_PERCENTILES[1]}_log10': float(p_hi),
                     'percentile_span_orders': float(span),
                     'min_log10': float(v.min()), 'max_log10': float(v.max()),
                     'full_range_orders': float(full)})

    ax.set_xlabel(f'log₁₀ mean {plural[:-1]} intensity', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'{plural.capitalize()}\n(intensity distribution)',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.20)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLOR[c], edgecolor='black', linewidth=0.4,
                     alpha=alpha) for c in COND_ORDER]
    labels = [f'{c}  {stats[c][0]:,} $\\cdot$ {stats[c][1]:.1f}/{stats[c][2]:.1f}'
              for c in COND_ORDER]
    leg = ax.legend(handles, labels, loc='upper right', frameon=False,
                    fontsize=7.5, handlelength=0.9, handleheight=0.9,
                    handletextpad=0.4, borderpad=0.2, labelspacing=0.35,
                    title=f'n $\\cdot$ orders P{DR_PERCENTILES[0]}-'
                          f'P{DR_PERCENTILES[1]}/full')
    leg.get_title().set_fontsize(6.5)
    leg.get_title().set_color('#555555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return rows


def _entity_sets(df, sample_info, level):
    id_col, _, _ = _LEVEL_SPEC[level]
    det = df[df['precursor_intensity'].notna()]
    return {cond: set(det.loc[det['run'].isin(
                set(sample_info.loc[sample_info['condition2'] == cond, 'run'])),
                id_col].dropna().unique())
            for cond in COND_ORDER}


def _set_overlap_on_ax(ax, df, sample_info, level):
    """Total per condition (light) with the all-condition core overlaid (solid), unique-to-condition annotated."""
    _, _, plural = _LEVEL_SPEC[level]
    sets = _entity_sets(df, sample_info, level)
    core_set = set.intersection(*sets.values())
    x = np.arange(len(COND_ORDER))
    max_y = max(len(s) for s in sets.values())
    rows = []
    for j, cond in enumerate(COND_ORDER):
        total = len(sets[cond])
        others = set.union(*(sets[c] for c in COND_ORDER if c != cond))
        unique = len(sets[cond] - others)
        ax.bar(x[j], total, width=0.6, color=COLOR[cond], alpha=0.45,
               edgecolor='darkgray', linewidth=1.0)
        ax.bar(x[j], len(core_set), width=0.6, color=COLOR[cond], alpha=1.0,
               edgecolor='black', linewidth=1.0)
        ax.text(x[j], total + max_y * LABEL_OFFSET, f'{total:,}', ha='center',
                va='bottom', fontsize=8)
        ax.text(x[j], len(core_set) - max_y * 0.015,
                f'{len(core_set):,}\nin all {len(COND_ORDER)}', ha='center',
                va='top', fontsize=8, color='white')
        ax.text(x[j], (len(core_set) + total) / 2, f'{unique:,}\nunique',
                ha='center', va='center', fontsize=7, color='#333333')
        rows.append({'panel': f'set_overlap_{level}', 'condition': cond,
                     'total': total, 'in_all_conditions': len(core_set),
                     'unique_to_condition': unique})
    _condition_xticklabels(ax, COND_ORDER)
    ax.set_ylabel(f'# {plural}', fontsize=10)
    ax.set_ylim(0, max_y * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.set_title(f'{plural.capitalize()}\n(light = total, solid = shared by all)',
                 fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return rows


def _pairwise_overlap_on_ax(ax, sets, level, plural):
    """Pairwise intersection matrix — the Venn's replacement past 3 conditions."""
    n = len(COND_ORDER)
    frac = np.zeros((n, n))
    rows = []
    for i, a in enumerate(COND_ORDER):
        for j, b in enumerate(COND_ORDER):
            inter = len(sets[a] & sets[b])
            frac[i, j] = inter / max(1, min(len(sets[a]), len(sets[b])))
            rows.append({'panel': f'pairwise_overlap_{level}', 'condition': a,
                         'other_condition': b, 'n_condition': len(sets[a]),
                         'n_other': len(sets[b]), 'intersection': inter,
                         'frac_of_smaller': float(frac[i, j])})
    ax.imshow(frac, cmap='Purples', vmin=0, vmax=1)
    fmt = lambda v: f'{v/1000:.1f}k' if v >= 10_000 else f'{v:,}'
    for i in range(n):
        for j in range(n):
            inter = int(round(frac[i, j] * min(len(sets[COND_ORDER[i]]),
                                               len(sets[COND_ORDER[j]]))))
            txt = (fmt(len(sets[COND_ORDER[i]])) if i == j
                   else f'{fmt(inter)}\n{frac[i, j]*100:.0f}%')
            ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                    color='white' if frac[i, j] > 0.6 else '#222222')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(COND_ORDER, fontsize=8, rotation=30, ha='right')
    ax.set_yticklabels(COND_ORDER, fontsize=8)
    ax.set_title(f'{plural.capitalize()}\n(shared, % of the smaller set)',
                 fontsize=11, fontweight='bold')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return rows


def _venn_on_ax(ax, df, sample_info, level):
    _, _, plural = _LEVEL_SPEC[level]
    sets = _entity_sets(df, sample_info, level)
    if len(COND_ORDER) > 3:
        return _pairwise_overlap_on_ax(ax, sets, level, plural)

    from matplotlib_venn import venn2, venn3
    venn_fn = venn2 if len(COND_ORDER) == 2 else venn3
    venn_fn([sets[c] for c in COND_ORDER], set_labels=COND_ORDER, ax=ax,
            set_colors=[COLOR[c] for c in COND_ORDER])
    ax.set_title(f'{plural.capitalize()}\n(overlap)',
                 fontsize=11, fontweight='bold')
    rows = []
    for cond in COND_ORDER:
        for ent in sets[cond]:
            rows.append({'panel': f'venn_{level}', 'condition': cond,
                         'entity': ent})
    return rows


def two_up_panels(df, sample_info):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    rows = (_rank_on_ax(axes[0], df, sample_info, 'precursor')
            + _rank_on_ax(axes[1], df, sample_info, 'protein_group',
                          point_size=4, alpha=0.45))
    fig.tight_layout()
    save(fig, 'rank_precursor_pg', pd.DataFrame(rows))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    rows = (_intensity_hist_on_ax(axes[0], df, sample_info, 'precursor')
            + _intensity_hist_on_ax(axes[1], df, sample_info, 'protein_group'))
    fig.tight_layout()
    save(fig, 'intensity_histogram_precursor_pg', pd.DataFrame(rows))

    if len(COND_ORDER) < 2:
        print('  overlap panel skipped: needs 2+ conditions '
              '(all quantities are cross-condition)')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        bar_rows = _set_overlap_on_ax(axes[0], df, sample_info, 'precursor')
        venn_rows = _venn_on_ax(axes[1], df, sample_info, 'protein_group')
        fig.tight_layout()
        save(fig, 'overlap_precursor_pg',
             pd.concat([pd.DataFrame(bar_rows), pd.DataFrame(venn_rows)],
                       ignore_index=True))


def pca_panel(df, sample_info):
    """PCA of protein groups, labelling only outlier runs."""
    from zenotof8600_analytics.plotting._style import _confidence_ellipse

    n_comp = int(min(8, max(2, len(sample_info) - 1)))
    _, _, res = core.plot_pca(df, sample_info, level='protein',
                              color_by='condition2', show_labels=False,
                              palette=COLOR, min_valid_fraction=0.7,
                              n_components=n_comp)
    plt.close('all')
    pca_df = res['transformed']
    ve = res['variance_explained'] * 100
    norms, thr = _pca_outliers(pca_df)

    fig, axes2d = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes2d.ravel()
    ax = axes[0]
    n_runs = len(pca_df)
    msize = 34 if n_runs <= 24 else 14
    for cond in COND_ORDER:
        sub = pca_df[pca_df['condition2'] == cond]
        if sub.empty:
            continue
        ax.scatter(sub['PC1'], sub['PC2'], s=msize, color=COLOR[cond],
                   alpha=0.85, edgecolors='black', linewidths=0.5, zorder=4)
        if len(sub) >= 3:
            try:
                import matplotlib.colors as mcolors
                fill = list(mcolors.to_rgba(COLOR[cond]))
                fill[3] = 0.18
                _confidence_ellipse(sub['PC1'].values, sub['PC2'].values, ax,
                                    n_std=float(np.sqrt(-2 * np.log(1 - 0.95))),
                                    facecolor=fill, edgecolor=COLOR[cond],
                                    linewidth=1.0, zorder=2)
            except (ValueError, np.linalg.LinAlgError):
                pass

    flagged, highlighted = [], []
    for _, r in pca_df.iterrows():
        run, tail = r['run'], r['run'].rsplit('_', 1)[-1]
        is_hi = _matches_highlight(run, PCA_HIGHLIGHT)
        is_out = bool(norms[run] > thr)
        if is_hi:
            highlighted.append(tail)
            ax.scatter([r['PC1']], [r['PC2']], s=msize * 1.6,
                       color=HIGHLIGHT_COLOR, edgecolors='black',
                       linewidths=0.6, zorder=6)
        elif is_out:
            flagged.append(tail)
        if is_hi or is_out:
            ax.annotate(tail, (r['PC1'], r['PC2']), fontsize=7,
                        xytext=(4, 3), textcoords='offset points', zorder=7,
                        color=HIGHLIGHT_COLOR if is_hi else COLOR[r['condition2']])

    ax.axhline(0, color='k', lw=0.5, alpha=0.5, zorder=1)
    ax.axvline(0, color='k', lw=0.5, alpha=0.5, zorder=1)
    ax.set_xlabel(f'PC1 ({ve[0]:.1f}% variance)', fontsize=10)
    ax.set_ylabel(f'PC2 ({ve[1]:.1f}% variance)', fontsize=10)
    ax.set_title('PCA - protein groups\n(outlier runs labelled)',
                 fontsize=11, fontweight='bold')
    _compact_legend(ax, COND_ORDER, loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    scree_rows = _scree_on_ax(axes[1], ve)
    factor_rows = _factor_heatmap_on_ax(axes[2], pca_df, ve)
    loading_rows = _pc1_loadings_on_ax(axes[3], res, df, pca_df)
    fig.tight_layout()

    src = pca_df.copy()
    src['panel'] = 'pca_scores'
    src['residual_from_condition_centroid'] = [norms[r] for r in src['run']]
    src['outlier_threshold'] = thr
    src['flagged_outlier'] = src['residual_from_condition_centroid'] > thr
    src['manually_highlighted'] = [_matches_highlight(r, PCA_HIGHLIGHT)
                                   for r in src['run']]
    save(fig, 'pca_protein_groups',
         pd.concat([src, pd.DataFrame(scree_rows), pd.DataFrame(factor_rows),
                    pd.DataFrame(loading_rows)], ignore_index=True))

    print(f'  PCA variance explained: PC1 {ve[0]:.1f}%, PC2 {ve[1]:.1f}%')
    if np.isfinite(thr):
        print(f'  outlier runs (residual > {thr:.1f}): '
              f'{", ".join(flagged) if flagged else "none"}')
    else:
        print(f'  outlier flagging skipped: no condition has '
              f'{MIN_REPS_FOR_FLAGGING}+ replicates')
    if highlighted:
        print(f'  manually highlighted: {", ".join(highlighted)}')


def library_panels(df, sample_info):

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    stats = []
    for ax, (level, label) in zip(axes, (('precursor', 'Precursor CV'),
                                         ('protein', 'Protein-group CV'))):
        _, _, cv_stats = core.plot_cv_violin(
            df, sample_info, level=level, group_col='condition2',
            group_order=COND_ORDER, palette=COLOR, ax=ax,
            as_percent=True, show_threshold=True, ylim=(0, 60),
            median_label_loc='inline', title=label)
        _unbold_axis_labels(ax)
        stats.append(cv_stats.assign(level=level))
    fig.tight_layout()
    save(fig, 'cv_violin_precursor_pg', pd.concat(stats, ignore_index=True))

    return None


def _mc_intensity_fractions(df, sample_info, mc_classes):
    """Per-run share of summed peptide intensity carried by each MC class."""
    det = df[df['precursor_intensity'].notna()]
    pep = (det.drop_duplicates(['run', 'peptide_id'])
              [['run', 'peptide_id', 'sequence', 'peptide_intensity']]
              .dropna(subset=['peptide_intensity']))
    pep = pep[pep['sequence'].str.len() > 0]
    cap = max(mc_classes)
    mc_lookup = {s: min(core.count_missed_cleavages(s), cap)
                 for s in pep['sequence'].unique()}
    pep['mc'] = pep['sequence'].map(mc_lookup)
    tab = (pep.pivot_table(index='run', columns='mc',
                           values='peptide_intensity', aggfunc='sum')
              .reindex(columns=mc_classes).fillna(0.0))
    return tab.div(tab.sum(axis=1), axis=0)


def _mc_bars_on_ax(ax, per_rep_by_cond, mc_classes, *, y_label, title):
    """Grouped MC bars, one group per MC class, with per-replicate points."""
    width = 0.8 / len(COND_ORDER)
    x = np.arange(len(mc_classes))
    rng = np.random.default_rng(0)
    rows = []
    for j, cond in enumerate(COND_ORDER):
        per_rep = per_rep_by_cond[cond]
        frac = per_rep.mean().to_numpy()
        centers = x + j * width - 0.4 + width / 2
        ax.bar(centers, frac, width=width, color=COLOR[cond],
               edgecolor='black', linewidth=0.8, label=cond)
        for xi, mc in zip(centers, mc_classes):
            vals = per_rep[mc].to_numpy()
            jitter = rng.uniform(-width * 0.22, width * 0.22, vals.size)
            ax.scatter(xi + jitter, vals, c='black', s=12, alpha=0.75,
                       zorder=5, edgecolors='white', linewidths=0.4)
        for xi, v in zip(centers, frac):
            ax.text(xi, v + 0.01, f'{v*100:.1f}', ha='center', va='bottom',
                    fontsize=7)
        for run, (_, r) in zip(per_rep.index, per_rep.iterrows()):
            for mc in mc_classes:
                rows.append({'condition': cond, 'run': run,
                             'series': 'per-run point',
                             'missed_cleavages': mc, 'fraction': float(r[mc])})
        for mc, v in zip(mc_classes, frac):
            rows.append({'condition': cond, 'run': 'MEAN',
                         'series': 'bar height', 'missed_cleavages': mc,
                         'fraction': float(v)})
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in mc_classes])
    ax.set_xlabel('Missed cleavages', fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(frameon=False, fontsize=8, loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return rows


def digestion_qc(per_run, df, sample_info):
    mc_cols = [c for c in per_run.columns if c.startswith('MC') and c[2:].isdigit()]
    mc_classes = [int(c[2:]) for c in mc_cols]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    rows = []

    by_count = {}
    for cond in COND_ORDER:
        sub = per_run[per_run['condition2'] == cond].set_index('run')
        per_rep = sub[mc_cols].div(sub[mc_cols].sum(axis=1), axis=0)
        per_rep.columns = mc_classes
        by_count[cond] = per_rep
    rows += [dict(panel='missed_cleavages_by_count', **r) for r in
             _mc_bars_on_ax(axes[0], by_count, mc_classes,
                            y_label='Fraction of peptides',
                            title='Missed cleavages\n(by peptide count)')]

    int_frac = _mc_intensity_fractions(df, sample_info, mc_classes)
    by_intensity = {}
    for cond in COND_ORDER:
        runs = sorted(sample_info.loc[sample_info['condition2'] == cond, 'run'])
        by_intensity[cond] = int_frac.reindex(runs)
    rows += [dict(panel='missed_cleavages_by_intensity', **r) for r in
             _mc_bars_on_ax(axes[1], by_intensity, mc_classes,
                            y_label='Fraction of peptide intensity',
                            title='Missed cleavages\n(by intensity)')]

    ax = axes[2]
    quant = df[df['precursor_intensity'].notna() & (df['precursor_intensity'] > 0)]
    bins = np.arange(5, 46) - 0.5
    lb = (bins[:-1] + bins[1:]) / 2

    def _length_profile(seqs):
        counts, _ = np.histogram(seqs.str.len(), bins=bins)
        total = counts.sum()
        return np.full(len(counts), np.nan) if total == 0 else counts / total

    for cond in COND_ORDER:
        runs = sorted(sample_info.loc[sample_info['condition2'] == cond, 'run'])
        for run in runs:
            seqs = quant.loc[quant['run'] == run, 'sequence'].drop_duplicates()
            prof = _length_profile(seqs)
            ax.plot(lb, prof, color=COLOR[cond], lw=0.6, alpha=0.35, zorder=1)
            for c, v in zip(lb, prof):
                rows.append({'panel': 'peptide_length', 'condition': cond,
                             'run': run, 'series': 'per-run profile',
                             'peptide_length': int(c), 'fraction': float(v)})
        seqs = quant.loc[quant['run'].isin(runs), 'sequence'].drop_duplicates()
        prof = _length_profile(seqs)
        ax.plot(lb, prof, color=COLOR[cond], lw=1.5, zorder=3,
                label=f'{cond} (median {int(seqs.str.len().median())})')
        for c, v in zip(lb, prof):
            rows.append({'panel': 'peptide_length', 'condition': cond,
                         'run': 'UNION', 'series': 'condition profile',
                         'peptide_length': int(c), 'fraction': float(v)})
    ax.set_xlabel('Peptide length [aa]', fontsize=10)
    ax.set_ylabel('Fraction of peptides', fontsize=10)
    ax.set_title('Peptide length', fontsize=11, fontweight='bold')
    ax.legend(frameon=False, fontsize=8, loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    save(fig, 'digestion_qc', pd.DataFrame(rows))


def run_analysis(df, sample_info):
    """Write the full panel set for an already-loaded experiment."""
    print('summary tables ...')
    per_run, agg = summary_tables(df, sample_info)
    with pd.option_context('display.width', 200, 'display.max_columns', 50):
        print(agg.to_string(index=False))

    print('CV screen ...')
    cv_sum = cv_screen(df, sample_info)
    with pd.option_context('display.width', 200, 'display.max_columns', 50):
        print(cv_sum.to_string(index=False))

    print('completeness ...')
    comp = completeness(df, sample_info)
    endpoints = comp[comp['k_replicates'].isin([1, comp['k_replicates'].max()])]
    print(endpoints.pivot_table(index=['condition', 'level'],
                                columns='k_replicates',
                                values='n_detected_in_at_least_k')
          .rename(columns=lambda c: f'>={c} reps').to_string())

    print('replicate variability ...')
    with pd.option_context('display.width', 200, 'display.max_columns', 50):
        print(replicate_variability(df, sample_info).to_string(index=False))

    print('PCA ...')
    if df['engine'].nunique() > 1:
        print('  skipped: df contains multiple search engines '
              f'({", ".join(sorted(df["engine"].unique()))}) — a PCA across '
              'engines separates by engine, not by condition')
    else:
        pca_panel(df, sample_info)

    print('two-up panels ...')
    two_up_panels(df, sample_info)

    print('library panels ...')
    library_panels(df, sample_info)

    print('digestion QC ...')
    digestion_qc(per_run, df, sample_info)

    print(f'\ndone -> {OUT_DIR}')


def main(**configure_kwargs):
    df, sample_info = load(**configure_kwargs)
    run_analysis(df, sample_info)
    return df, sample_info


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('data_root', nargs='?', default=None,
                    help='folder with one subfolder per condition '
                         f'(default: {DATA_ROOT})')
    ap.add_argument('--out-dir', default=None, help='default <data_root>/analysis')
    ap.add_argument('--conditions', default=None,
                    help='comma-separated subfolders to include, in display '
                         'order; write LABEL=FOLDER to show a shorter name than '
                         'the folder in every panel (e.g. "HEK=HEK_30SPD_100vw12ms")')
    ap.add_argument('--labels', default=None,
                    help='comma-separated LABEL=CONDITION display renames, '
                         'applied after --split-runs so split labels can be '
                         'renamed too')
    ap.add_argument('--split-runs', default=None,
                    help='split each search output into conditions by run name: '
                         'a plate field (plate_group, plate_quadrant, plate_row, '
                         'plate_col, plate_well), a regex with one capture group '
                         r'(e.g. "_(Q[1-4])_"), or "LABEL=REGEX,LABEL=REGEX"')
    ap.add_argument('--split-order', default=None,
                    help='comma-separated display order of the SPLIT condition '
                         'labels (--conditions applies to the folders)')
    ap.add_argument('--outlier-k', type=float, default=None,
                    help='MAD multiples for outlier cutoffs (default 3.0)')
    a = ap.parse_args()

    order, labels = None, {}
    if a.conditions:
        for part in a.conditions.split(','):
            label, sep, folder = part.partition('=')
            if sep:
                labels[folder] = label
            else:
                folder = label
            order = (order or []) + [folder]
    if a.labels:
        for part in a.labels.split(','):
            label, sep, cond = part.partition('=')
            if not sep:
                ap.error(f'--labels expects LABEL=CONDITION, got {part!r}')
            labels[cond] = label

    split = a.split_runs
    if split and '=' in split:
        split = dict(part.split('=', 1) for part in split.split(','))
    split_order = a.split_order.split(',') if a.split_order else None
    main(data_root=a.data_root, out_dir=a.out_dir,
         include_only=order, cond_order=split_order or (None if split else order),
         split_runs=split, outlier_k=a.outlier_k, labels=labels or None)
