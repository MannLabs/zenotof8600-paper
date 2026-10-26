"""Figure 5: ZT Scan DIA against conventional DIA across throughputs."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import io
import os
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

import zenotof8600_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_E386 = os.path.join(cfg.DATA_ROOT, 'H032_E386')
DATA_E392 = os.path.join(cfg.DATA_ROOT, 'H032_E392')
OUT = cfg.panels(__file__)

sys.path.insert(0, os.path.dirname(HERE))
import figure_style as fs

LETTER = {'ids_pg': 'b', 'rate': 'c',
          'fractions_pep': 'd', 'fractions_pg': 'e'}

SPECIES = ['Ecoli', 'Yeast', 'Human']
SPECIES_LABEL = {'Ecoli': 'E. coli', 'Yeast': 'Yeast', 'Human': 'Human'}
PROTEOME_SIZE = {'Ecoli': 4_403, 'Yeast': 6_060, 'Human': 20_435}

THROUGHPUTS = [200, 300, 500]
METHODS = ['ZT', 'DIA']
METHOD_LABEL = {'ZT': 'ZT Scan', 'DIA': 'DIA'}
SPECIES_COLOR = {'Human': core.PALETTE_SINGLE[1],
                 'Yeast': core.PALETTE_SINGLE[0],
                 'Ecoli': core.PALETTE_SINGLE[2]}

IDS_LIGHTEN = 0.68
DIA_MIX_TARGET = '#2F3A46'
DIA_MIX = 0.45

FRACTION_LIGHTEN_MAX = 0.48

LEVELS = {
    'protein': dict(id_col='protein_group', int_col='pg_intensity',
                    y_label='# protein groups', short='protein groups'),
    'peptide': dict(id_col='peptide_id', int_col='peptide_intensity',
                    y_label='# peptides', short='peptides'),
    'precursor': dict(id_col='precursor_id', int_col='precursor_intensity',
                      y_label='# precursors', short='precursors'),
}
CV_THRESHOLD = 0.20
MIN_VALUES_FOR_CV = 3

BAR_IN = 0.12
BAR_UNIT_IN = 0.15
GROUP_GAP = 0.9

ROW2_AXES_W_IN = 1.68

POINT_FRAC = 0.30
POINT_MAX_PT = 6.5
POINT_JITTER = 0.22

RT_RANGE_PCT = 99.0

MIN_PER_SAMPLE = {spd: 1440.0 / spd for spd in THROUGHPUTS}

THOUSANDS = FuncFormatter(lambda v, _: f'{v:,.0f}')

def method_shade(hue, method):
    """The species hue as that scan mode draws it."""
    return hue if method == 'ZT' else fs.mix(hue, DIA_MIX_TARGET, DIA_MIX)


ARM_RE = re.compile(r'(\d+)SPD_(DIA|ZT)_([A-Za-z]+)$')
FRACTION_RE = re.compile(r'([A-Za-z]+)_(\d+)fractions$')


def _runs_in(features_csv):
    """Run names, from the per-run `<run> m/z` columns of a PEAKS export."""
    cols = pd.read_csv(features_csv, nrows=0).columns
    return [c[:-len(' m/z')] for c in cols
            if c.endswith(' m/z') and c != 'm/z']


def _load_folder(folder, condition1, condition2):
    features = os.path.join(folder, 'lfq.dia.features.csv')
    df, sample_info = core.load_experiments([{
        'path': features,
        'protein_path': os.path.join(folder, 'lfq.dia.proteins.csv'),
        'file_tags': _runs_in(features),
        'condition1': condition1, 'condition2': condition2}])
    sample_info['replicate'] = sample_info.groupby('condition2').cumcount() + 1
    return df, sample_info


def per_run_counts(df, runs, level):
    """That level's identifications in each run, house definition."""
    spec = LEVELS[level]
    quant = df[df['precursor_intensity'].notna() & (df['precursor_intensity'] > 0)]
    sub = quant[quant[spec['int_col']].notna() & (quant[spec['int_col']] > 0)]
    return [int(sub.loc[sub['run'] == r, spec['id_col']].nunique()) for r in runs]


def union_count(df, runs, level):
    """Distinct entities of that level identified in ANY of the arm's runs."""
    spec = LEVELS[level]
    quant = df[df['precursor_intensity'].notna() & (df['precursor_intensity'] > 0)]
    sub = quant[quant[spec['int_col']].notna() & (quant[spec['int_col']] > 0)
                & quant['run'].isin(runs)]
    return int(sub[spec['id_col']].nunique())


def cv20_count(df, sample_info, level):
    """Entities whose CV across the arm's runs is under 20%."""
    tab = core._compute_cv_table(df, sample_info, level=level,
                                 group_col='condition2',
                                 min_values_for_cv=MIN_VALUES_FOR_CV)
    return int((tab['cv'] < CV_THRESHOLD).sum()), tab['cv'].to_numpy()


def elution_window(df, runs):
    """Central `RT_RANGE_PCT` of each run's identified precursor retention times."""
    quant = df[df['precursor_intensity'].notna()]
    tail = (100.0 - RT_RANGE_PCT) / 2.0
    spans, edges = [], []
    for r in runs:
        rt = quant.loc[quant['run'] == r, 'rt'].dropna().to_numpy()
        if rt.size < 100:
            spans.append(np.nan)
            edges.append((np.nan, np.nan))
            continue
        lo, hi = np.percentile(rt, [tail, 100.0 - tail])
        spans.append(float(hi - lo))
        edges.append((float(lo), float(hi)))
    return spans, edges


def load_e386():
    """One row per arm, carrying every count the panels need."""
    arms = []
    for name in sorted(os.listdir(DATA_E386)):
        if name.endswith('_repeat'):
            continue
        m = ARM_RE.fullmatch(name)
        if not m:
            continue
        spd, method, species = int(m.group(1)), m.group(2), m.group(3)
        df, sample_info = _load_folder(
            os.path.join(DATA_E386, name), f'{spd} SPD {species}', method)
        runs = list(sample_info['run'])
        row = {'arm': name, 'spd': spd, 'method': method, 'species': species,
               'n_runs': len(runs)}
        for level in LEVELS:
            counts = per_run_counts(df, runs, level)
            cv20, _ = cv20_count(df, sample_info, level)
            row[f'{level}_per_run'] = counts
            row[f'{level}_mean'] = float(np.mean(counts))
            row[f'{level}_union'] = union_count(df, runs, level)
            row[f'{level}_sd'] = float(np.std(counts, ddof=1))
            row[f'{level}_cv20'] = cv20
        spans, edges = elution_window(df, runs)
        row['rt_span_per_run'] = spans
        row['rt_span_min'] = float(np.nanmean(spans))
        row['rt_lo_min'] = float(np.nanmean([e[0] for e in edges]))
        row['rt_hi_min'] = float(np.nanmean([e[1] for e in edges]))
        arms.append(row)
        print(f'  {name:22s} n={len(runs)}  '
              f'PG {row["protein_mean"]:>6,.0f} (<20% {row["protein_cv20"]:>6,.0f})  '
              f'pep {row["peptide_mean"]:>7,.0f} (<20% {row["peptide_cv20"]:>7,.0f})  '
              f'RT {row["rt_lo_min"]:.2f}-{row["rt_hi_min"]:.2f} min',
              flush=True)
    return pd.DataFrame(arms)


def _feature_rows(folder):
    """Data rows in a PEAKS features table, without parsing it."""
    path = os.path.join(folder, 'lfq.dia.features.csv')
    if not os.path.exists(path):
        return 0
    with io.open(path, encoding='utf-8', errors='replace') as f:
        return max(sum(1 for _ in f) - 1, 0)


def load_e392():
    """Per-run detection for every fraction set, long format."""
    frames = []
    for name in sorted(os.listdir(DATA_E392)):
        m = FRACTION_RE.fullmatch(name)
        if not m:
            continue
        species, declared = m.group(1), int(m.group(2))
        folder = os.path.join(DATA_E392, name)

        if _feature_rows(folder) == 0:
            print(f'  SKIPPING {name}: the search has ZERO feature rows -- an '
                  f'empty result, not a measurement', flush=True)
            continue

        df, sample_info = _load_folder(folder, f'{species} fractionated', name)
        runs = list(sample_info['run'])

        if len(runs) != declared:
            print(f'  NOTE {name}: {len(runs)} runs against {declared} in the '
                  f'folder name -- drawn as {len(runs)} fractions',
                  flush=True)
        frames.append(df.assign(species=species, n_fractions=len(runs),
                                fraction_set=declared))
        print(f'  {name:22s} {len(runs)} fractions, {len(df):,} rows',
              flush=True)
    if not frames:
        raise ValueError(f'no usable fraction set under {DATA_E392}')
    return pd.concat(frames, ignore_index=True)


def _bar_positions():
    """x for every (species, throughput, method) slot, plus tick anchors."""
    slots, pair_x, group_x = {}, {}, {}
    for si, species in enumerate(SPECIES):
        base = si * (len(THROUGHPUTS) * len(METHODS) + GROUP_GAP)
        for ti, spd in enumerate(THROUGHPUTS):
            for mi, method in enumerate(METHODS):
                slots[(species, spd, method)] = base + ti * len(METHODS) + mi
            pair_x[(species, spd)] = base + ti * len(METHODS) + 0.5
        group_x[species] = base + (len(THROUGHPUTS) * len(METHODS) - 1) / 2
    return slots, pair_x, group_x


def panel_ids(arms, level, key):
    """b, c -- identifications with the CV < 20% subset, by species and throughput."""
    spec = LEVELS[level]
    slots, pair_x, group_x = _bar_positions()
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    rng = np.random.default_rng(0)
    point_size = core.replicate_point_size(BAR_IN)
    rows, dots = [], []

    for _, r in arms.iterrows():
        x = slots[(r['species'], r['spd'], r['method'])]
        hue = SPECIES_COLOR[r['species']]
        ax.bar(x, r[f'{level}_union'], color=fs.lighten(hue, IDS_LIGHTEN),
               edgecolor='darkgray', linewidth=0.8, zorder=2)
        ax.bar(x, r[f'{level}_cv20'], color=method_shade(hue, r['method']),
               edgecolor='black', linewidth=0.8, zorder=3)
        per_run = r[f'{level}_per_run']
        dots.append(ax.scatter(
            x + rng.uniform(-POINT_JITTER, POINT_JITTER, len(per_run)),
            per_run,
            s=point_size, color='black', alpha=0.75, linewidth=0.3,
            edgecolor='white', zorder=5))
        rows.append({'species': SPECIES_LABEL[r['species']], 'spd': r['spd'],
                     'method': METHOD_LABEL[r['method']],
                     'x': x, 'union': r[f'{level}_union'],
                     'mean_per_run': r[f'{level}_mean'],
                     'sd_per_run': r[f'{level}_sd'],
                     'cv20': r[f'{level}_cv20'],
                     'per_run': ';'.join(str(v) for v in per_run)})

    ax.set_xticks([pair_x[(s, t)] for s in SPECIES for t in THROUGHPUTS])
    ax.set_xticklabels([str(t) for _ in SPECIES for t in THROUGHPUTS])
    all_x = list(slots.values())
    ax.set_xlim(min(all_x) - 0.8, max(all_x) + 0.8)
    ax.set_ylabel(spec['y_label'], fontsize=fs.FONTSIZE)
    ax.set_xlabel('Samples per day', fontsize=fs.FONTSIZE, labelpad=18)
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)

    trans = ax.get_xaxis_transform()
    for species in SPECIES:
        ax.text(group_x[species], -0.115, SPECIES_LABEL[species],
                transform=trans, ha='center', va='top',
                fontsize=fs.FONTSIZE, clip_on=False)

    grey = '#7F7F7F'
    handles = [Patch(facecolor=method_shade(grey, m),
                     edgecolor='black', linewidth=0.8,
                     label=f'{METHOD_LABEL[m]}, < 20% CV') for m in METHODS]
    handles.append(Patch(facecolor=fs.lighten(grey, IDS_LIGHTEN),
                         edgecolor='darkgray', linewidth=0.8,
                         label='all IDs'))
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False, fontsize=fs.LEGEND_FONTSIZE,
              handlelength=1.1, handletextpad=0.5, columnspacing=1.6,
              borderpad=0.15)

    fig.tight_layout()
    fs.set_bar_geometry_inches(fig, ax, bar_in=BAR_IN, unit_in=BAR_UNIT_IN,
                               h_in=fs.AXES_H_IN)
    core.scale_replicate_points(ax, dots)
    prefix = f'figure5{LETTER[key]}_' if key in LETTER else ''
    name = (f'{prefix}'
            f'{"protein_group" if level == "protein" else level}_ids')
    fs.save_panel(fig, OUT, name, pd.DataFrame(rows), dpi=300)
    plt.close(fig)


RATE_METHODS = ('ZT',)


def panel_rate(arms):
    """c -- protein groups per minute of elution, against throughput."""
    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 3.4))
    rows = []
    for species in SPECIES:
        for method in RATE_METHODS:
            sub = (arms[(arms['species'] == species) &
                        (arms['method'] == method)]
                   .set_index('spd').reindex(THROUGHPUTS).reset_index())
            rate = sub['protein_mean'] / sub['rt_span_min']
            if rate.isna().all():
                continue
            shade = method_shade(SPECIES_COLOR[species], method)
            ax.plot(sub['spd'], rate, color=shade, linewidth=1.6,
                    linestyle='-' if method == 'ZT' else '--',
                    marker='o', markersize=6,
                    markerfacecolor=(shade if method == 'ZT' else 'white'),
                    markeredgecolor=shade, markeredgewidth=1.3, zorder=3)
            for _, r in sub.dropna(subset=['protein_mean']).iterrows():
                rows.append({'species': SPECIES_LABEL[species],
                             'method': METHOD_LABEL[method], 'spd': r['spd'],
                             'protein_groups': r['protein_mean'],
                             'elution_span_min': r['rt_span_min'],
                             'protein_groups_per_min':
                                 r['protein_mean'] / r['rt_span_min']})

    ax.set_xlabel('Samples per day', fontsize=fs.FONTSIZE)
    ax.set_ylabel('Protein groups eluting per minute', fontsize=fs.FONTSIZE)
    ax.set_xticks(THROUGHPUTS)
    ax.set_xlim(160, 540)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)

    method_handles = [
        plt.Line2D([], [], color=method_shade('#7F7F7F', m), linewidth=1.6,
                   linestyle='-' if m == 'ZT' else '--', marker='o',
                   markersize=6, markeredgecolor=method_shade('#7F7F7F', m),
                   markeredgewidth=1.3,
                   markerfacecolor=(method_shade('#7F7F7F', m) if m == 'ZT'
                                    else 'white'),
                   label=METHOD_LABEL[m]) for m in RATE_METHODS]
    species_handles = [
        plt.Line2D([], [], color=SPECIES_COLOR[s], linewidth=2.2,
                   label=SPECIES_LABEL[s]) for s in SPECIES]
    opts = dict(loc='lower center', frameon=False,
                fontsize=fs.LEGEND_FONTSIZE, handlelength=1.3,
                handletextpad=0.3, borderpad=0.0, columnspacing=0.5,
                labelspacing=0.25)
    if len(RATE_METHODS) > 1:
        low = ax.legend(handles=method_handles, ncol=len(method_handles),
                        bbox_to_anchor=(0.5, 1.02), **opts)
        ax.add_artist(low)
        fig.canvas.draw()
        h = low.get_window_extent().transformed(ax.transAxes.inverted()).height
    else:
        h = 0.0
    ax.legend(handles=species_handles, ncol=len(species_handles),
              bbox_to_anchor=(0.5, 1.02 + h + (0.015 if h else 0.0)), **opts)

    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=ROW2_AXES_W_IN,
                            h_in=fs.AXES_H_IN)
    fs.save_panel(fig, OUT, f'figure5{LETTER["rate"]}_depth_per_elution_minute',
                  pd.DataFrame(rows), dpi=300)
    plt.close(fig)


def _natural_key(name):
    """Sort C1, C2, ..."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', name)]


def cumulative_depth(long_df):
    """Union of protein groups and peptides after each successive fraction."""
    long_df = long_df[long_df['precursor_intensity'].notna()]
    rows = []
    for (species, n_fractions), sub in long_df.groupby(
            ['species', 'n_fractions'], sort=True):
        runs = sorted(sub['run'].unique(), key=_natural_key)
        seen = {'protein_group': set(), 'peptide_id': set(),
                'precursor_id': set()}
        for i, run in enumerate(runs, start=1):
            r = sub[sub['run'] == run]
            for col in seen:
                seen[col] |= set(r[col].dropna().unique())
            rows.append({
                'species': species, 'n_fractions': n_fractions,
                'set_label': f'{SPECIES_LABEL.get(species, species)}, '
                             f'{n_fractions} fr',
                'fraction_index': i, 'run': run,
                'cumulative_protein_groups': len(seen['protein_group']),
                'cumulative_peptides': len(seen['peptide_id']),
                'cumulative_precursors': len(seen['precursor_id']),
                'instrument_min': i * MIN_PER_SAMPLE[200]})
    return pd.DataFrame(rows)


FRACTION_STYLE = {6: ':', 12: '--', 23: '-', 24: '-'}
FRACTION_STYLE_LABEL = {6: '6 fr', 12: '12 fr', 24: '24 fr'}


def _drawn_sets(cum, sets=None):
    """The rows of `cum` belonging to a drawn fraction set."""
    keep = [n - d for n in (sets or FRACTION_SETS_DRAWN) for d in (0, 1)]
    return cum[cum['n_fractions'].isin(keep)]


def _fraction_labels_and_colors(cum):
    """Set labels in species order; hue is the species, nothing is shaded."""
    ordered = cum.sort_values(['species', 'n_fractions'])
    labels = [lab for sp in SPECIES for lab in
              dict.fromkeys(ordered.loc[ordered['species'] == sp,
                                        'set_label'])]
    colors = {lab: SPECIES_COLOR[ordered.loc[ordered['set_label'] == lab,
                                             'species'].iloc[0]]
              for lab in labels}
    return labels, colors


FRACTION_PANELS = [
    ('fractions_pep', 'cumulative_peptides', 'peptide_mean',
     '# peptides, cumulative', 'peptides'),
    ('fractions_pg', 'cumulative_protein_groups', 'protein_mean',
     '# protein groups, cumulative', 'protein_groups'),
]

FRACTION_SETS_DRAWN = (24,)


def _fraction_panel(cum, single_shot, *, key, value_col, ref_col, y_label,
                    stem, sets=None, out=None, name=None, axes_w_in=None,
                    shot_color=None, legend_colspacing=0.35):
    """One level's cumulative depth against number of fractions, 200 SPD."""
    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 3.4))
    cum = _drawn_sets(cum, sets)
    labels, colors = _fraction_labels_and_colors(cum)

    for lab in labels:
        sub = cum[cum['set_label'] == lab].sort_values('fraction_index')
        n = int(sub['n_fractions'].iloc[0])
        ax.plot(sub['fraction_index'], sub[value_col], color=colors[lab],
                linewidth=1.7, linestyle=FRACTION_STYLE.get(n, '-'),
                marker='o', markersize=3.4, markeredgecolor='white',
                markeredgewidth=0.4, zorder=3)

    for species, ref in single_shot.items():
        ax.axhline(ref[ref_col], color=shot_color or SPECIES_COLOR[species],
                   linewidth=1.2,
                   linestyle=(0, (6, 3)), zorder=1)

    ax.set_xlabel('Fractions acquired', fontsize=fs.FONTSIZE)
    ax.set_ylabel(y_label, fontsize=fs.FONTSIZE)
    ax.set_ylim(0, cum[value_col].max() * 1.05)
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.tick_params(labelsize=fs.FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.xaxis.label.set_fontweight('normal')
    ax.yaxis.label.set_fontweight('normal')

    species_h = [plt.Line2D([], [], color=SPECIES_COLOR[sp], linewidth=1.7,
                            label=SPECIES_LABEL[sp])
                 for sp in SPECIES if (cum['species'] == sp).any()]
    depth_h = [plt.Line2D([], [], color='#9A9A9A', linewidth=1.7,
                          linestyle=FRACTION_STYLE[n],
                          label=FRACTION_STYLE_LABEL[n])
               for n in (6, 12, 24)
               if cum['n_fractions'].isin([n, n - 1]).any()]
    depth_h.append(plt.Line2D([], [], color=fs.LEGEND_DARK, linewidth=1.2,
                              linestyle=(0, (6, 3)), label='1 shot'))
    top = ax.secondary_xaxis(
        'top', functions=(lambda v: v * MIN_PER_SAMPLE[200],
                          lambda v: v / MIN_PER_SAMPLE[200]))
    top.set_xlabel('Instrument time [min]', fontsize=fs.FONTSIZE)
    top.tick_params(labelsize=fs.FONTSIZE)

    fig.canvas.draw()
    inv = ax.transAxes.inverted()
    base = max(top.get_window_extent().transformed(inv).y1,
               top.xaxis.label.get_window_extent().transformed(inv).y1) + 0.03
    opts = dict(loc='lower center', frameon=False,
                fontsize=fs.LEGEND_FONTSIZE, handlelength=0.9,
                handletextpad=0.25, borderpad=0.0, columnspacing=legend_colspacing,
                labelspacing=0.25)
    low = ax.legend(handles=depth_h, ncol=len(depth_h),
                    bbox_to_anchor=(0.5, base), **opts)
    ax.add_artist(low)
    fig.canvas.draw()
    h = low.get_window_extent().transformed(inv).height
    ax.legend(handles=species_h, ncol=len(species_h),
              bbox_to_anchor=(0.5, base + h + 0.015), **opts)

    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=axes_w_in or ROW2_AXES_W_IN,
                            h_in=fs.AXES_H_IN)
    source = cum[['species', 'n_fractions', 'set_label', 'fraction_index',
                  'run', 'instrument_min', value_col]]
    fs.save_panel(fig, out or OUT,
                  name or f'figure5{LETTER[key]}_fractionation_{stem}',
                  source, dpi=300)
    plt.close(fig)


def panel_fractions(cum, single_shot, *, wanted=None):
    """d, e -- one cumulative-depth panel per level."""
    for key, value_col, ref_col, y_label, stem in FRACTION_PANELS:
        if wanted is not None and LETTER[key] not in wanted:
            continue
        print(f'panel {LETTER[key]} ...')
        _fraction_panel(cum, single_shot, key=key, value_col=value_col,
                        ref_col=ref_col, y_label=y_label, stem=stem)


def report(arms, cum, single_shot):
    for level, name in [('protein', 'protein groups'), ('peptide', 'peptides')]:
        wide = arms.pivot_table(index=['species', 'spd'], columns='method',
                                values=[f'{level}_union', f'{level}_cv20'])
        print(f'\n{name}, ZT Scan over DIA [%]   (IDs | CV<20%)')
        for species in SPECIES:
            cells = []
            for spd in THROUGHPUTS:
                try:
                    r = wide.loc[(species, spd)]
                except KeyError:
                    cells.append('    --     ')
                    continue
                ids = r[(f'{level}_union', 'ZT')] / r[(f'{level}_union', 'DIA')]
                q20 = r[(f'{level}_cv20', 'ZT')] / r[(f'{level}_cv20', 'DIA')]
                if not np.isfinite(ids) or not np.isfinite(q20):
                    cells.append('    --     ')
                else:
                    cells.append(f'{(ids - 1) * 100:+6.1f} |{(q20 - 1) * 100:+6.1f}')
            print(f'  {SPECIES_LABEL[species]:8s} ' + '  '.join(cells))
        print('  (columns: ' +
              ', '.join(f'{t} SPD' for t in THROUGHPUTS) + ')')

    print('\nelution window and depth per minute')
    for _, r in arms.sort_values(['species', 'spd', 'method']).iterrows():
        print(f'  {r["arm"]:22s} {r["rt_lo_min"]:5.2f}-{r["rt_hi_min"]:5.2f} min '
              f'= {r["rt_span_min"]:4.2f} min   '
              f'{r["protein_mean"] / r["rt_span_min"]:7,.0f} PG/min')

    print('\nfractionation, 200 SPD')
    for lab, sub in cum.groupby('set_label'):
        sub = sub.sort_values('fraction_index')
        last = sub.iloc[-1]
        species = sub['species'].iloc[0]
        ref = single_shot[species]['protein_mean']
        ref_pep = single_shot[species]['peptide_mean']
        total = last['cumulative_protein_groups']
        total_pep = last['cumulative_peptides']
        n = int(last['n_fractions'])
        k = int(sub.loc[sub['cumulative_protein_groups'] >= 0.9 * total,
                        'fraction_index'].iloc[0])
        k_pep = int(sub.loc[sub['cumulative_peptides'] >= 0.9 * total_pep,
                            'fraction_index'].iloc[0])
        print(f'  {lab:18s} {last["instrument_min"]:>5.1f} min')
        print(f'      protein groups {total:>7,.0f}  '
              f'{total / ref:.2f}x the single shot ({ref:>7,.0f})  '
              f'90% after {k:>2d} of {n}')
        print(f'      peptides       {total_pep:>7,.0f}  '
              f'{total_pep / ref_pep:.2f}x the single shot ({ref_pep:>7,.0f})  '
              f'90% after {k_pep:>2d} of {n}')

    print('\ngain against proteome size (for the text, not a panel)')
    pts = []
    for species in SPECIES:
        for spd in THROUGHPUTS:
            sub = arms[(arms['species'] == species) & (arms['spd'] == spd)]
            if set(sub['method']) != set(METHODS):
                continue
            zt = float(sub.loc[sub['method'] == 'ZT', 'protein_union'].iloc[0])
            dia = float(sub.loc[sub['method'] == 'DIA', 'protein_union'].iloc[0])
            pts.append((PROTEOME_SIZE[species], (zt / dia - 1) * 100))
    x = np.log10([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    print(f'  r = {np.corrcoef(x, y)[0, 1]:.3f} over {len(pts)} paired arms '
          f'(log proteome size vs protein-group gain)')


def main(*letters):
    """Build all panels, or only the letters asked for."""
    wanted = set(letters) or set(LETTER.values())
    unknown = wanted - set(LETTER.values())
    if unknown:
        raise SystemExit(f'unknown panel(s) {sorted(unknown)}; this figure '
                         f'has {sorted(set(LETTER.values()))}')
    fs.init()
    fs.start_build(script=__file__, inputs=[DATA_E386, DATA_E392], out_dir=OUT)
    if wanted != set(LETTER.values()):
        print(f'building only: {" ".join(sorted(wanted))}')
    core.set_replicate_point_scale(frac=POINT_FRAC, max_pt=POINT_MAX_PT)

    cache = {}

    def arms():
        if 'arms' not in cache:
            print(f'\nloading {DATA_E386} ...')
            cache['arms'] = load_e386()
            print(f'  {len(cache["arms"])} arms')
        return cache['arms']

    def fractions():
        """Cumulative depth, plus the single-shot level each curve is read against -- which comes from E386, so d and e need both datasets."""
        if 'cum' not in cache:
            print(f'\nloading {DATA_E392} ...')
            cum = cumulative_depth(load_e392())
            single_shot = {}
            for species in cum['species'].unique():
                ref = arms()[(arms()['species'] == species)
                             & (arms()['spd'] == 200)
                             & (arms()['method'] == 'ZT')]
                if ref.empty:
                    raise ValueError(f'no 200 SPD ZT {species} arm to anchor '
                                     f'the single-shot level')
                single_shot[species] = {c: float(ref[c].iloc[0])
                                        for c in ('protein_mean',
                                                  'peptide_mean')}
            cache['cum'] = (cum, single_shot)
        return cache['cum']

    if LETTER['ids_pg'] in wanted:
        print(f'\npanel {LETTER["ids_pg"]} ...')
        panel_ids(arms(), 'protein', 'ids_pg')
    if LETTER['rate'] in wanted:
        print(f'panel {LETTER["rate"]} ...')
        panel_rate(arms())
    if {LETTER['fractions_pep'], LETTER['fractions_pg']} & wanted:
        cum, single_shot = fractions()
        panel_fractions(cum, single_shot, wanted=wanted)

    if {'arms', 'cum'} <= set(cache):
        report(arms(), *fractions())
    else:
        print('\n(summary table skipped: it needs both datasets, and this '
              'build loaded only what its panels required)')
    fs.report_printed_sizes()
    fs.report_stale(OUT)


if __name__ == '__main__':
    main(*sys.argv[1:])
