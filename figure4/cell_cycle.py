"""Figure 4 — cell-cycle trajectory of the single cells, H032_E388."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import zenotof8600_analytics as core

import single_cell as sc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = cfg.panels(__file__)

sys.path.insert(0, os.path.dirname(HERE))
import figure_style as fs

S_GENES = [
    'MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2',
    'MCM6', 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', 'CENPU', 'HELLS',
    'RFC2', 'RPA2', 'NASP', 'RAD51AP1', 'GMNN', 'WDR76', 'SLBP', 'CCNE2',
    'UBR7', 'POLD3', 'MSH2', 'ATAD2', 'RAD51', 'RRM2', 'CDC45', 'CDC6',
    'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 'USP1', 'CLSPN', 'POLA1',
    'CHAF1B', 'BRIP1', 'E2F8',
]

G2M_GENES = [
    'HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80',
    'CKS2', 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'FAM64A',
    'PIMREG', 'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11',
    'ANP32E', 'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'CDCA3', 'HN1', 'JPT1',
    'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', 'DLGAP5',
    'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 'ANLN',
    'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5',
    'CENPA',
]

MIN_DETECTED_FRAC = 0.50
MIN_MARKERS_PER_CELL = 3
N_BINS = 25
CTRL_SIZE = 50
OUTLIER_SD = 4.0

PSEUDOTIME_CMAP = 'YlOrRd'

LETTER = 'c'

PHASE_ORDER = ['G1', 'S', 'G2/M']


def save(fig, ax, name, source_df, *, w_in=1.62):
    fig.tight_layout()
    fs.set_axes_size_inches(fig, ax, w_in=w_in, h_in=fs.AXES_H_IN)
    return fs.save_panel(fig, OUT, name, source_df, dpi=300)


def protein_matrix():
    """cells x proteins, log2, restricted to proteins seen in >=50% of cells."""
    runs = sc.bp.run_names(sc.FEATURES)
    experiments = [{'path': sc.FEATURES,
                    'file_tags': [r for r in runs if c['match'] in r],
                    'condition1': 'H032_E388', 'condition2': c['label']}
                   for c in sc.CONDITIONS]
    df, _ = core.load_experiments(experiments)

    ids, _ = sc.apply_exclusions(sc.load())
    keep = set(ids.loc[ids['included']
                       & (ids['condition2'] == sc.SINGLE_CELL), 'run'])
    print(f'  {len(keep)} single cells kept by the box-panel exclusions')

    d = df[df['run'].isin(keep)].dropna(subset=['genes'])
    d = d[d['precursor_intensity'] > 0]
    pg = (d.groupby(['run', 'protein_group'])
          .agg(intensity=('precursor_intensity', 'sum'),
               genes=('genes', 'first')).reset_index())

    mat = pg.pivot(index='run', columns='protein_group', values='intensity')
    gene_of = (pg.drop_duplicates('protein_group')
               .set_index('protein_group')['genes'])
    frac = mat.notna().mean(axis=0)
    mat = mat.loc[:, frac >= MIN_DETECTED_FRAC]
    print(f'  {mat.shape[1]:,} of {len(frac):,} protein groups quantified in '
          f'>={MIN_DETECTED_FRAC:.0%} of cells')
    return np.log2(mat), gene_of.reindex(mat.columns)


def score_genes(mat, columns, rng):
    """scanpy's `score_genes`, reimplemented (see the module docstring)."""
    mean_abundance = mat.mean(axis=0, skipna=True)
    ranks = mean_abundance.rank(method='min')
    bins = np.ceil(ranks / len(ranks) * N_BINS).astype(int)

    controls = set()
    for col in columns:
        same_bin = bins.index[(bins == bins[col])].tolist()
        same_bin = [c for c in same_bin if c not in columns]
        if not same_bin:
            continue
        take = min(CTRL_SIZE, len(same_bin))
        controls.update(rng.choice(same_bin, size=take, replace=False))
    controls = sorted(controls)

    n_markers = mat[list(columns)].notna().sum(axis=1)
    score = (mat[list(columns)].mean(axis=1, skipna=True)
             - mat[controls].mean(axis=1, skipna=True))
    score[n_markers < MIN_MARKERS_PER_CELL] = np.nan
    return score, n_markers, len(controls)


def cell_cycle(mat, gene_of):
    """S and G2/M scores, discrete phase and angular pseudotime per cell."""
    upper = gene_of.fillna('').str.upper()

    def columns_for(genes):
        want = set(genes)
        hit = upper.apply(lambda s: bool(want & {t.strip()
                                                 for t in s.split(';')}))
        return list(gene_of.index[hit])

    rng = np.random.default_rng(42)
    s_cols, g_cols = columns_for(S_GENES), columns_for(G2M_GENES)
    print(f'  markers found: S {len(s_cols)}, G2/M {len(g_cols)}')
    if len(s_cols) < MIN_MARKERS_PER_CELL or len(g_cols) < MIN_MARKERS_PER_CELL:
        raise ValueError('too few markers quantified to score a phase')

    s, n_s, n_ctrl_s = score_genes(mat, s_cols, rng)
    g, n_g, n_ctrl_g = score_genes(mat, g_cols, rng)
    print(f'  controls drawn: S {n_ctrl_s}, G2/M {n_ctrl_g}')

    out = pd.DataFrame({'run': mat.index, 's_score': s.values,
                        'g2m_score': g.values, 'n_s_markers': n_s.values,
                        'n_g2m_markers': n_g.values})
    out = out.dropna(subset=['s_score', 'g2m_score'])

    keep = np.ones(len(out), bool)
    for col in ('s_score', 'g2m_score'):
        z = (out[col] - out[col].mean()) / out[col].std()
        keep &= z.abs() <= OUTLIER_SD
    if (~keep).any():
        print(f'  {int((~keep).sum())} scoring outlier(s) dropped '
              f'(>{OUTLIER_SD} s.d.)')
    out = out[keep].reset_index(drop=True)

    out['phase'] = np.where(
        (out['s_score'] <= 0) & (out['g2m_score'] <= 0), 'G1',
        np.where(out['s_score'] >= out['g2m_score'], 'S', 'G2/M'))

    theta = np.arctan2(out['g2m_score'] - out['g2m_score'].mean(),
                       out['s_score'] - out['s_score'].mean()) % (2 * np.pi)
    g1 = theta[out['phase'] == 'G1']
    origin = float(np.median(g1)) if len(g1) else 0.0
    out['pseudotime_rad'] = (theta - origin) % (2 * np.pi)
    print('  ' + '  '.join(f'{p} {int((out["phase"] == p).sum())}'
                           for p in PHASE_ORDER))
    return out.sort_values('pseudotime_rad').reset_index(drop=True)


def panel_pseudotime(cc):
    """The preprint's 2G: the (S, G2/M) plane coloured by pseudotime."""
    fig, ax = plt.subplots(figsize=(fs.PANEL_W_IN, 4))
    sm = ax.scatter(cc['s_score'], cc['g2m_score'], c=cc['pseudotime_rad'],
                    cmap=PSEUDOTIME_CMAP, vmin=0, vmax=2 * np.pi, s=80,
                    edgecolor='black', linewidth=0.5, zorder=3)
    ax.axhline(0, color='0.65', lw=0.8, ls=':', zorder=0)
    ax.axvline(0, color='0.65', lw=0.8, ls=':', zorder=0)
    ax.set_xlabel('S score (protein)', fontsize=fs.FONTSIZE)
    ax.set_ylabel('G2/M score (protein)', fontsize=fs.FONTSIZE)
    ax.tick_params(labelsize=fs.FONTSIZE)
    cb = fig.colorbar(sm, ax=ax, orientation='horizontal', location='top',
                      fraction=0.055, pad=0.04,
                      ticks=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    cb.ax.set_xticklabels(['0', '$\\pi/2$', '$\\pi$', '3$\\pi$/2', '2$\\pi$'])
    cb.set_label('pseudotime (rad)', fontsize=fs.FONTSIZE)
    cb.ax.tick_params(labelsize=fs.FONTSIZE)
    fs.tidy(ax)
    save(fig, ax, f'figure4{LETTER}_cell_cycle_pseudotime', cc)
    plt.close(fig)


def main():
    fs.init()
    fs.start_build(script=__file__, inputs=[os.path.join(cfg.DATA_ROOT, 'H032_E388')], out_dir=OUT)
    mat, gene_of = protein_matrix()
    print('\ncell-cycle scoring ...')
    cc = cell_cycle(mat, gene_of)
    print('\npanels ...')
    panel_pseudotime(cc)
    fs.report_printed_sizes()
    fs.report_stale(OUT, prefix='figure')


if __name__ == '__main__':
    main()
