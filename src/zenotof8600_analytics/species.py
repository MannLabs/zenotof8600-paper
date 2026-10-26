"""Mixed-species benchmarks — species assignment, composition and ratio recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .plotting._style import PALETTE_SINGLE, _resolve_panel_size

__all__ = [
    'SPECIES_SUFFIXES',
    'assign_species',
    'sum_precursors_to_protein',
    'species_ratio_table',
    'species_ratio_accuracy',
    'species_ratio_precision',
    'plot_expected_composition',
    'plot_species_counts',
    'plot_species_ratio',
    'plot_ratio_accuracy',
    'plot_ratio_precision',
    'plot_species_cv_ecdf',
]

SPECIES_SUFFIXES = {
    'HUMAN': 'Human',
    'YEAST': 'Yeast',
    'ECOLI': 'E. coli',
    'MOUSE': 'Mouse',
    'ARATH': 'A. thaliana',
    'CAEEL': 'C. elegans',
    'DROME': 'D. melanogaster',
}

AMBIGUOUS = 'Ambiguous'
UNKNOWN = 'Unknown'


def assign_species(df, *, col='protein_names', suffixes=None, sep=';',
                   out_col='species'):
    """Label each row with the organism of its protein group."""
    suffixes = SPECIES_SUFFIXES if suffixes is None else suffixes
    if col not in df.columns:
        raise KeyError(
            f'{col!r} not in df; species come from the UniProt entry name '
            f'(available: {sorted(df.columns)[:12]}...)')

    unique = pd.Series(df[col].dropna().unique())
    lookup = {}
    for value in unique:
        found = set()
        for name in str(value).split(sep):
            name = name.strip()
            suffix = name.rsplit('_', 1)[-1].upper() if '_' in name else ''
            found.add(suffixes.get(suffix, UNKNOWN))
        found.discard(UNKNOWN)
        if len(found) == 1:
            lookup[value] = found.pop()
        elif len(found) > 1:
            lookup[value] = AMBIGUOUS
        else:
            lookup[value] = UNKNOWN

    out = df.copy()
    out[out_col] = out[col].map(lookup).fillna(UNKNOWN)
    return out


def _condition_runs(sample_info, group_col):
    return {c: set(g['run']) for c, g in sample_info.groupby(group_col)}


def _pg_matrix(df, runs, value_col='pg_intensity', id_col='protein_group'):
    """One row per protein group, one column per run, intensities > 0."""
    sub = df[df['run'].isin(runs)].dropna(subset=[id_col, value_col])
    sub = sub[sub[value_col] > 0]
    return sub.pivot_table(index=id_col, columns='run', values=value_col,
                           aggfunc='first')


def sum_precursors_to_protein(df, runs, *, id_col='protein_group',
                              value_col='precursor_intensity',
                              precursor_col='precursor_id',
                              min_completeness=1.0):
    """Protein quant by summing precursor intensities, on complete precursors."""
    sub = df[df['run'].isin(runs)].dropna(subset=[id_col, value_col,
                                                  precursor_col])
    sub = sub[sub[value_col] > 0]
    wide = sub.pivot_table(index=precursor_col, columns='run',
                           values=value_col, aggfunc='max')
    wide = wide.reindex(columns=list(runs))
    keep = wide.notna().sum(axis=1) >= min_completeness * len(runs)
    wide = wide[keep]

    pg = (sub.drop_duplicates(precursor_col)
          .set_index(precursor_col)[id_col].reindex(wide.index))
    out = wide.groupby(pg).sum(min_count=1)
    return out.where(out > 0)


def species_ratio_table(df, sample_info, condition_a, condition_b, *,
                        group_col='condition2', value_col='pg_intensity',
                        id_col='protein_group', species_col='species',
                        min_completeness=1.0, max_cv=None,
                        drop_ambiguous=True, quant='column',
                        precursor_col='precursor_id',
                        normalize_species=None):
    """Per-protein measured log2 ratio between two conditions."""
    runs = _condition_runs(sample_info, group_col)
    for cond in (condition_a, condition_b):
        if cond not in runs:
            raise ValueError(f'condition {cond!r} not in '
                             f'sample_info[{group_col!r}] '
                             f'(available: {sorted(runs)})')

    species = (df[[id_col, species_col]].drop_duplicates()
               .set_index(id_col)[species_col])

    if quant == 'sum_precursors':
        all_runs = list(runs[condition_a]) + list(runs[condition_b])
        combined = sum_precursors_to_protein(
            df, all_runs, id_col=id_col, value_col=value_col,
            precursor_col=precursor_col, min_completeness=min_completeness)
        matrices = {tag: combined[list(runs[cond])]
                    for tag, cond in (('a', condition_a), ('b', condition_b))}
    elif quant == 'column':
        matrices = {tag: _pg_matrix(df, runs[cond], value_col, id_col)
                    for tag, cond in (('a', condition_a), ('b', condition_b))}
    else:
        raise ValueError("quant must be 'column' or 'sum_precursors', "
                         f'got {quant!r}')

    if normalize_species is not None:
        wide = pd.concat(matrices.values(), axis=1)
        log2_all = np.log2(wide)
        is_ref = log2_all.index.map(species).to_series(
            index=log2_all.index).eq(normalize_species)
        if not is_ref.any():
            raise ValueError(
                f'normalize_species={normalize_species!r} matches no protein '
                'group; nothing to normalise against')
        shift = log2_all.loc[is_ref.to_numpy()].median(axis=0)
        matrices = {tag: 2.0 ** (np.log2(m).sub(shift[m.columns], axis=1))
                    for tag, m in matrices.items()}

    stats = {}
    for tag, cond in (('a', condition_a), ('b', condition_b)):
        mat = matrices[tag]
        n_runs = len(runs[cond])
        log2 = np.log2(mat)
        stats[tag] = pd.DataFrame({
            f'mean_log2_{tag}': log2.mean(axis=1),
            f'cv_{tag}': mat.std(axis=1, ddof=1) / mat.mean(axis=1),
            f'sd_log2_{tag}': log2.std(axis=1, ddof=1),
            f'n_{tag}': mat.notna().sum(axis=1),
            f'completeness_{tag}': mat.notna().sum(axis=1) / n_runs,
        })

    out = stats['a'].join(stats['b'], how='inner')
    out['log2_ratio'] = out['mean_log2_a'] - out['mean_log2_b']
    out['log2_intensity'] = (out['mean_log2_a'] + out['mean_log2_b']) / 2
    out = out.reset_index().rename(columns={'index': id_col})
    out[species_col] = out[id_col].map(species).fillna(UNKNOWN)

    n_before = len(out)
    keep = ((out['completeness_a'] >= min_completeness)
            & (out['completeness_b'] >= min_completeness))
    if max_cv is not None:
        keep &= (out['cv_a'].fillna(0) <= max_cv) & (out['cv_b'].fillna(0) <= max_cv)
    if drop_ambiguous:
        keep &= ~out[species_col].isin([AMBIGUOUS, UNKNOWN])
    out = out[keep].reset_index(drop=True)
    n_ambiguous = int((~keep).sum())
    out.attrs['n_before_filter'] = n_before
    out.attrs['n_dropped'] = n_ambiguous
    return out


def expected_log2_ratios(composition_a, composition_b, *, reference=None,
                         offset=0.0):
    """Nominal log2 ratio per species from two mixing designs."""
    species = sorted(set(composition_a) | set(composition_b))
    out = {}
    for s in species:
        a, b = composition_a.get(s, 0.0), composition_b.get(s, 0.0)
        out[s] = np.log2(a / b) + offset if a > 0 and b > 0 else np.nan
    return out


def species_ratio_accuracy(df, sample_info, pairs, compositions, *,
                           species_order=None, verbose=True, **table_kwargs):
    """Measured against expected log2 ratio, over several condition pairs."""
    rows = []
    for cond_a, cond_b in pairs:
        table = species_ratio_table(df, sample_info, cond_a, cond_b,
                                    **table_kwargs)
        species = (species_order if species_order is not None
                   else sorted(table['species'].unique()))
        expected = expected_log2_ratios(compositions[cond_a],
                                        compositions[cond_b])
        for s in species:
            v = table.loc[table['species'] == s, 'log2_ratio'].dropna()
            if v.empty or not np.isfinite(expected.get(s, np.nan)):
                continue
            rows.append({'condition_a': cond_a, 'condition_b': cond_b,
                         'species': s, 'expected': float(expected[s]),
                         'observed': float(v.median()),
                         'observed_mean': float(v.mean()),
                         'sd': float(v.std(ddof=1)), 'n': int(len(v))})
        if verbose:
            print(f'  {cond_a} vs {cond_b}: ' + '  '.join(
                f'{r["species"]} {r["observed"]:+.3f}/{r["expected"]:+.3f}'
                for r in rows if r['condition_a'] == cond_a
                and r['condition_b'] == cond_b))

    accuracy = pd.DataFrame(rows)
    fit = _fit_line(accuracy['expected'].to_numpy(),
                    accuracy['observed'].to_numpy())
    if verbose:
        print(f'  compression slope {fit["slope"]:.3f}, '
              f'intercept {fit["intercept"]:+.3f}, R² {fit["r_squared"]:.4f}')
    return accuracy, fit


def _fit_line(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {'slope': float(slope), 'intercept': float(intercept),
            'r_squared': 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            'rms_residual': float(np.sqrt(ss_res / len(y))),
            'n_points': int(len(y))}


def plot_ratio_accuracy(accuracy, fit, *, correct=False, species_order=None,
                        palette=None, figsize=None, title=None,
                        label_fontsize=10, tick_fontsize=10,
                        legend_fontsize=8, ax=None):
    """Measured against expected log2 ratio, with the identity line."""
    import matplotlib.pyplot as plt

    acc = accuracy.copy()
    slope = fit['slope']
    if correct:
        acc['observed'] = acc['observed'] / slope
        shown = _fit_line(acc['expected'].to_numpy(), acc['observed'].to_numpy())
    else:
        shown = fit

    species = (list(species_order) if species_order is not None
               else sorted(acc['species'].unique()))
    palette = palette or {s: PALETTE_SINGLE[i % len(PALETTE_SINGLE)]
                          for i, s in enumerate(species)}

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_resolve_panel_size(figsize))
    else:
        fig = ax.figure

    lo = min(acc['expected'].min(), acc['observed'].min()) - 0.4
    hi = max(acc['expected'].max(), acc['observed'].max()) + 0.4
    ax.plot([lo, hi], [lo, hi], ls='--', color='#888888', lw=1, zorder=1,
            label='y = x')
    for s in species:
        sub = acc[acc['species'] == s]
        ax.scatter(sub['expected'], sub['observed'], s=55, color=palette[s],
                   edgecolor='black', linewidth=0.6, zorder=3, label=s)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Expected log₂ ratio', fontsize=label_fontsize)
    ax.set_ylabel(('Measured log₂ ratio, corrected' if correct
                   else 'Measured log₂ ratio'), fontsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=label_fontsize + 1, fontweight='bold')
    ax.tick_params(labelsize=tick_fontsize)

    note = (f'compression {slope:.3f}\n'
            f'residual RMS {shown["rms_residual"]:.3f}\n'
            f'R² {shown["r_squared"]:.4f}') if correct else (
        f'slope {shown["slope"]:.3f}\n'
        f'intercept {shown["intercept"]:+.3f}\n'
        f'R² {shown["r_squared"]:.4f}')
    ax.text(0.97, 0.03, note, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=legend_fontsize, family='monospace')
    ax.legend(fontsize=legend_fontsize, frameon=False, loc='upper left')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if created:
        fig.tight_layout()
    acc['fit_slope'] = shown['slope']
    acc['fit_intercept'] = shown['intercept']
    acc['compression_slope'] = slope
    return fig, ax, acc


def plot_species_cv_ecdf(df, sample_info, condition, *, group_col='condition2',
                         species_col='species', id_col='protein_group',
                         value_col='pg_intensity', min_values=3,
                         species_order=None, palette=None, figsize=None,
                         cv_threshold=20.0, xlim=(0, 40), title=None,
                         label_fontsize=10, tick_fontsize=10,
                         legend_fontsize=8, ax=None):
    """ECDF of per-protein CV within one condition, one curve per species."""
    import matplotlib.pyplot as plt

    runs = sample_info.loc[sample_info[group_col] == condition, 'run']
    if runs.empty:
        raise ValueError(f'condition {condition!r} not in '
                         f'sample_info[{group_col!r}]')
    mat = _pg_matrix(df, set(runs), value_col, id_col)
    mat = mat[mat.notna().sum(axis=1) >= min_values]
    cv = (mat.std(axis=1, ddof=1) / mat.mean(axis=1) * 100).rename('cv_pct')

    species = (df[[id_col, species_col]].drop_duplicates()
               .set_index(id_col)[species_col])
    frame = pd.DataFrame({'cv_pct': cv,
                          'species': cv.index.map(species)}).dropna()
    frame = frame[~frame['species'].isin([AMBIGUOUS, UNKNOWN])]

    order = (list(species_order) if species_order is not None
             else sorted(frame['species'].unique()))
    palette = palette or {s: PALETTE_SINGLE[i % len(PALETTE_SINGLE)]
                          for i, s in enumerate(order)}

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_resolve_panel_size(figsize))
    else:
        fig = ax.figure

    rows = []
    for s in order:
        v = np.sort(frame.loc[frame['species'] == s, 'cv_pct'].to_numpy())
        if v.size == 0:
            continue
        y = np.arange(1, v.size + 1) / v.size
        median = float(np.median(v))
        below = float((v < cv_threshold).mean())
        ax.plot(v, y, color=palette[s], lw=1.8,
                label=f'{s} ({median:.1f}%)')
        rows.extend({'species': s, 'cv_pct': float(c),
                     'cumulative_fraction': float(f)} for c, f in zip(v, y))
        print(f'    {s:<8} median CV {median:.1f}%   '
              f'{below:.1%} below {cv_threshold:g}%   n = {v.size:,}')

    ax.axvline(cv_threshold, ls=':', color='#d62728', lw=1.2, alpha=0.8)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_yticks(np.arange(0, 1.01, 0.05), minor=True)
    ax.set_xlabel('Protein CV [%]', fontsize=label_fontsize)
    ax.set_ylabel('Cumulative fraction', fontsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=label_fontsize + 1, fontweight='bold')
    ax.tick_params(labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize, frameon=False, loc='lower right',
              title='median CV', title_fontsize=legend_fontsize)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if created:
        fig.tight_layout()
    return fig, ax, pd.DataFrame(rows)


def plot_expected_composition(compositions, *, species_order=None,
                              condition_order=None, palette=None,
                              figsize=None, title=None,
                              y_label='Percentage of protein amounts',
                              x_label='Sample', label_fontsize=10,
                              tick_fontsize=10, legend_fontsize=8,
                              value_fontsize=8, ax=None):
    """Stacked bars of the designed species composition per condition."""
    conditions = (list(compositions) if condition_order is None
                  else list(condition_order))
    species = species_order or sorted(
        {s for c in compositions.values() for s in c}, reverse=True)
    palette = palette or {s: PALETTE_SINGLE[i % len(PALETTE_SINGLE)]
                          for i, s in enumerate(species)}
    if isinstance(palette, (list, tuple)):
        palette = {s: palette[i % len(palette)] for i, s in enumerate(species)}

    import matplotlib.pyplot as plt
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_resolve_panel_size(figsize))
    else:
        fig = ax.figure

    rows = []
    bottom = np.zeros(len(conditions))
    x = np.arange(len(conditions))
    for s in species:
        vals = np.array([compositions[c].get(s, 0.0) for c in conditions],
                        dtype=float)
        ax.bar(x, vals, bottom=bottom, width=0.7, label=s,
               color=palette[s], edgecolor='none')
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0:
                ax.text(xi, b + v / 2, f'{v:g}', ha='center', va='center',
                        fontsize=value_fontsize)
            rows.append({'condition': conditions[xi], 'species': s,
                         'percentage': float(v)})
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right',
                       fontsize=tick_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylim(0, max(bottom) * 1.02)
    if title:
        ax.set_title(title, fontsize=label_fontsize + 1, fontweight='bold')
    ax.legend(fontsize=legend_fontsize, frameon=False, loc='upper left',
              bbox_to_anchor=(1.02, 1), borderaxespad=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if created:
        fig.tight_layout()
    return fig, ax, pd.DataFrame(rows)


def plot_species_counts(df, sample_info, *, group_col='condition2',
                        species_col='species', id_col='protein_group',
                        value_col='pg_intensity', condition_order=None,
                        species_order=None, palette=None, figsize=None,
                        title=None, y_label='Number of proteins',
                        label_fontsize=10, tick_fontsize=10,
                        legend_fontsize=8, value_fontsize=7,
                        count_basis='detected', evidence_col='precursor_intensity',
                        drop_ambiguous=True, ax=None):
    """Stacked bars of identifications per species per condition."""
    import matplotlib.pyplot as plt

    if count_basis not in ('detected', 'quantified'):
        raise ValueError("count_basis must be 'detected' or 'quantified', "
                         f'got {count_basis!r}')
    basis_col = evidence_col if count_basis == 'detected' else value_col
    work = df.dropna(subset=[basis_col])
    work = work[work[basis_col] > 0]
    if drop_ambiguous:
        work = work[~work[species_col].isin([AMBIGUOUS, UNKNOWN])]
    work = work.merge(sample_info[['run', group_col]], on='run', how='left')

    counts = (work.groupby([group_col, species_col])[id_col]
              .nunique().unstack(fill_value=0))
    conditions = list(condition_order) if condition_order is not None else list(counts.index)
    species = list(species_order) if species_order is not None else list(counts.columns)
    counts = counts.reindex(index=conditions, columns=species, fill_value=0)

    palette = palette or {s: PALETTE_SINGLE[i % len(PALETTE_SINGLE)]
                          for i, s in enumerate(species)}
    if isinstance(palette, (list, tuple)):
        palette = {s: palette[i % len(palette)] for i, s in enumerate(species)}

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_resolve_panel_size(figsize))
    else:
        fig = ax.figure

    x = np.arange(len(conditions))
    bottom = np.zeros(len(conditions))
    rows = []
    for s in species:
        vals = counts[s].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, width=0.7, label=s, color=palette[s],
               edgecolor='none')
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0:
                ax.text(xi, b + v / 2, f'{int(v):,}', ha='center', va='center',
                        fontsize=value_fontsize, color='white')
            rows.append({'condition': conditions[xi], 'species': s,
                         'n': int(v)})
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right',
                       fontsize=tick_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    if title:
        ax.set_title(title, fontsize=label_fontsize + 1, fontweight='bold')
    ax.legend(fontsize=legend_fontsize, frameon=False, loc='upper left',
              bbox_to_anchor=(1.02, 1), borderaxespad=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if created:
        fig.tight_layout()
    return fig, ax, pd.DataFrame(rows)


def plot_species_ratio(df, sample_info, condition_a, condition_b, *,
                       composition_a=None, composition_b=None,
                       expected_ratios=None, reference_species='Human',
                       group_col='condition2', species_col='species',
                       id_col='protein_group', value_col='pg_intensity',
                       min_completeness=1.0, max_cv=None,
                       species_order=None, palette=None,
                       figsize=(12, 4), width_ratios=(6, 1, 1),
                       ylim=None, point_size=6, point_alpha=0.5,
                       label_fontsize=10, tick_fontsize=10,
                       legend_fontsize=8, rasterize_over=7000,
                       **table_kwargs):
    """Ratio-recovery panel: MA-style scatter, marginal density, and boxplots."""
    import matplotlib.pyplot as plt
    from scipy import stats as sps

    ratio_df = species_ratio_table(
        df, sample_info, condition_a, condition_b, group_col=group_col,
        value_col=value_col, id_col=id_col, species_col=species_col,
        min_completeness=min_completeness, max_cv=max_cv, **table_kwargs)

    species = (list(species_order) if species_order is not None
               else list(ratio_df[species_col].value_counts().index))
    palette = palette or {s: PALETTE_SINGLE[i % len(PALETTE_SINGLE)]
                          for i, s in enumerate(species)}
    if isinstance(palette, (list, tuple)):
        palette = {s: palette[i % len(palette)] for i, s in enumerate(species)}

    offset = 0.0
    if expected_ratios is None:
        if composition_a is None or composition_b is None:
            raise ValueError('pass either expected_ratios= or both '
                             'composition_a= and composition_b=')
        if reference_species is not None:
            ref = ratio_df.loc[ratio_df[species_col] == reference_species,
                               'log2_ratio']
            if ref.empty:
                raise ValueError(
                    f'reference_species={reference_species!r} has no proteins '
                    'after filtering; pass reference_species=None to skip the '
                    'loading-offset correction')
            offset = float(ref.median())
        expected_ratios = expected_log2_ratios(composition_a, composition_b,
                                               offset=offset)

    fig, axes = plt.subplots(
        1, 3, figsize=figsize, sharey=True,
        gridspec_kw={'width_ratios': list(width_ratios), 'wspace': 0.06})
    ax_scatter, ax_density, ax_box = axes

    for s in species:
        sub = ratio_df[ratio_df[species_col] == s]
        coll = ax_scatter.scatter(
            sub['log2_intensity'], sub['log2_ratio'], s=point_size,
            color=palette[s], alpha=point_alpha, edgecolor='none', label=s)
        if len(sub) > rasterize_over:
            coll.set_rasterized(True)

    for s in species:
        exp = expected_ratios.get(s, np.nan)
        if np.isfinite(exp):
            ax_scatter.axhline(exp, color='#444444', ls='--', lw=0.8, zorder=0)
            n = int((ratio_df[species_col] == s).sum())
            ax_scatter.annotate(
                f'n = {n:,}', xy=(0.62, exp), xycoords=('axes fraction', 'data'),
                va='bottom', ha='left', fontsize=legend_fontsize)

    rows = []
    for s in species:
        v = ratio_df.loc[ratio_df[species_col] == s, 'log2_ratio'].dropna()
        if v.empty:
            continue
        if len(v) < 3 or np.isclose(v.std(ddof=0), 0):
            ax_density.axhline(float(v.iloc[0]), color=palette[s], lw=1.0)
        else:
            grid = np.linspace(v.min(), v.max(), 512)
            dens = sps.gaussian_kde(v)(grid)
            ax_density.plot(dens, grid, color=palette[s], lw=1.0)
            ax_density.fill_betweenx(grid, 0, dens, color=palette[s], alpha=0.25)
        rows.append({'species': s, 'n': int(len(v)), 'median': float(v.median()),
                     'mean': float(v.mean()), 'sd': float(v.std(ddof=1)),
                     'expected_log2_ratio': float(expected_ratios.get(s, np.nan)),
                     'loading_offset_log2': offset})

    stats = pd.DataFrame(rows)
    for _, r in stats.iterrows():
        ax_density.annotate(
            f'{r["sd"]:.2f}', xy=(0.95, r['expected_log2_ratio']),
            xycoords=('axes fraction', 'data'), ha='right', va='bottom',
            fontsize=legend_fontsize, color=palette[r['species']])
    ax_density.set_title('s.d.', fontsize=label_fontsize, loc='right')

    box_data = [ratio_df.loc[ratio_df[species_col] == s, 'log2_ratio'].dropna()
                for s in species]
    bp = ax_box.boxplot(box_data, positions=range(len(species)), widths=0.6,
                        patch_artist=True, showcaps=True,
                        flierprops=dict(marker='o', markersize=1.5,
                                        markerfacecolor='#333333',
                                        markeredgecolor='none', alpha=0.4))
    for patch, s in zip(bp['boxes'], species):
        patch.set_facecolor(palette[s])
        patch.set_edgecolor('black')
    for median in bp['medians']:
        median.set_color('black')
    for s in species:
        exp = expected_ratios.get(s, np.nan)
        if np.isfinite(exp):
            ax_box.axhline(exp, color='#d62728', ls='--', lw=0.7, zorder=0)

    ax_scatter.set_xlabel('log₂(intensity)', fontsize=label_fontsize)
    ax_scatter.set_ylabel('log₂(ratio)', fontsize=label_fontsize)
    ax_scatter.legend(fontsize=legend_fontsize, frameon=False, loc='upper left',
                      markerscale=3)
    ax_density.set_xlabel('Density', fontsize=label_fontsize)
    ax_box.set_xticks(range(len(species)))
    ax_box.set_xticklabels([s[0] for s in species], fontsize=tick_fontsize)
    if ylim is not None:
        ax_scatter.set_ylim(*ylim)
    for ax in axes:
        ax.tick_params(labelsize=tick_fontsize)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    for ax in (ax_density, ax_box):
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)

    return fig, axes, stats, ratio_df


def _load_percent(composition, species):
    """A species' share of one condition, as a percentage."""
    value = composition.get(species, np.nan)
    total = sum(v for v in composition.values() if np.isfinite(v))
    if not np.isfinite(value) or total <= 0:
        return np.nan
    return float(value) / total * 100.0


def species_ratio_precision(df, sample_info, pairs, compositions, *,
                            species_order=None, verbose=True, **table_kwargs):
    """Precision of a mixed-species ratio measurement, over several pairs."""
    from .stats import sigma_iqr

    rows, cvs = [], {}
    for cond_a, cond_b in pairs:
        table = species_ratio_table(df, sample_info, cond_a, cond_b,
                                    **table_kwargs)
        species = (species_order if species_order is not None
                   else sorted(table['species'].unique()))
        expected = expected_log2_ratios(compositions[cond_a],
                                        compositions[cond_b])
        for s in species:
            sub = table[table['species'] == s]
            v = sub['log2_ratio'].dropna()
            if v.empty or not np.isfinite(expected.get(s, np.nan)):
                continue
            spread = sigma_iqr(v)
            rows.append({'condition_a': cond_a, 'condition_b': cond_b,
                         'species': s, 'expected': float(expected[s]),
                         'sigma_iqr_log2': float(spread),
                         'error_pct': float((2.0 ** spread - 1.0) * 100.0),
                         'n': int(len(v))})
            for cond, col in ((cond_a, 'cv_a'), (cond_b, 'cv_b')):
                cv = sub[col].dropna() if col in sub.columns else pd.Series(
                    dtype=float)
                if not cv.empty:
                    cvs.setdefault((cond, s), []).append(float(cv.median()))

    between_proteins = pd.DataFrame(
        rows, columns=['condition_a', 'condition_b', 'species', 'expected',
                       'sigma_iqr_log2', 'error_pct', 'n'])
    protein_cv = pd.DataFrame(
        [{'condition': cond, 'species': s,
          'load_pct': _load_percent(compositions[cond], s),
          'protein_cv_pct': float(np.mean(cvs[(cond, s)])) * 100.0}
         for cond, s in sorted(cvs, key=lambda k: (str(k[1]), str(k[0])))],
        columns=['condition', 'species', 'load_pct', 'protein_cv_pct'])

    if verbose and not between_proteins.empty:
        print(f'  ratio error between proteins: median '
              f'{between_proteins["error_pct"].median():.1f}%')
        print(f'  protein quant CV:             median '
              f'{protein_cv["protein_cv_pct"].median():.1f}%')
    return between_proteins, protein_cv


def plot_ratio_precision(between_proteins, protein_cv, *, species_order=None,
                         palette=None, figsize=(8, 4), threshold=20.0,
                         title=None, label_fontsize=10, tick_fontsize=10,
                         legend_fontsize=9.5, axes=None):
    """Two-panel precision summary for a mixed-species benchmark."""
    import matplotlib.pyplot as plt

    from .plotting._style import _hide_top_right_spines

    order = (list(species_order) if species_order is not None
             else sorted(set(between_proteins['species'])
                         | set(protein_cv['species'])))
    palette = PALETTE_SINGLE if palette is None else palette
    if isinstance(palette, (list, tuple)):
        palette = {s: palette[i % len(palette)] for i, s in enumerate(order)}

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)
    fig = axes[0].figure

    marker = dict(marker='o', markersize=8, markeredgecolor='black',
                  markeredgewidth=0.8, zorder=3)
    for s in order:
        b = between_proteins[between_proteins['species'] == s]
        if not b.empty:
            axes[0].plot(b['expected'], b['error_pct'], linestyle='none',
                         color=palette[s], label=str(s), **marker)
        c = protein_cv[protein_cv['species'] == s].sort_values('load_pct')
        if not c.empty:
            axes[1].plot(c['load_pct'], c['protein_cv_pct'], linestyle='-',
                         lw=1.4, color=palette[s], label=str(s), **marker)

    axes[0].axvline(0, color='black', lw=0.8, alpha=0.4, zorder=0)
    axes[0].set_xlabel('Expected log$_2$ ratio', fontsize=label_fontsize)
    axes[0].set_ylabel('Precision [%]', fontsize=label_fontsize)
    axes[0].set_title('Ratio error between proteins of one species',
                      fontsize=label_fontsize)
    axes[0].set_ylim(0, None)
    if len(order):
        axes[0].legend(frameon=False, fontsize=legend_fontsize,
                       loc='upper center', title='species',
                       title_fontsize=legend_fontsize)
    axes[1].set_xlabel('Species share of mixture [%]', fontsize=label_fontsize)
    axes[1].set_title('Protein quant CV within one condition',
                      fontsize=label_fontsize)

    for ax in axes:
        if threshold:
            ax.axhline(threshold, ls=':', color='#d62728', lw=1.2, zorder=0)
        ax.tick_params(labelsize=tick_fontsize)
        _hide_top_right_spines(ax)
    if title:
        fig.suptitle(title, fontsize=label_fontsize + 1, y=1.0)
    fig.tight_layout()

    source = pd.concat([
        between_proteins.assign(panel='ratio_error_between_proteins'),
        protein_cv.assign(panel='protein_quant_cv')], ignore_index=True)
    return fig, axes, source
