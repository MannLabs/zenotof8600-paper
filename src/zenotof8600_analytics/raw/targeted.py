"""Targeted quantification from raw files — fragment XICs, areas, calibration."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .extraction import (XIC_FWHM_FRACTION, extract_xic, integrate_peak,
                         measure_mz_shift_ppm, pick_peak, points_in_window,
                         resolve_mz_tol)

__all__ = [
    'SIL_MODIFICATIONS',
    'FRAGMENT_TYPES',
    'EXTRA_FRAGMENT_TYPES',
    'register_extra_fragment_types',
    'register_sil_modifications',
    'split_charged_frag_type',
    'target_fragments',
    'y_ions_above_precursor',
    'top_n_fragments',
    'summed_fragment_xic',
    'quantify_targets',
    'calibration_fit',
]

SIL_MODIFICATIONS = {
    'Label:13C(4)15N(1)@T': {'composition': 'C(-4)13C(4)N(-1)15N(1)'},
    'Label:13C(3)15N(1)@S': {'composition': 'C(-3)13C(3)N(-1)15N(1)'},
    'Label:13C(3)15N(1)@A': {'composition': 'C(-3)13C(3)N(-1)15N(1)'},
    'Label:13C(5)15N(1)@V': {'composition': 'C(-5)13C(5)N(-1)15N(1)'},
    'Label:13C(6)15N(1)@L': {'composition': 'C(-6)13C(6)N(-1)15N(1)'},
    'Label:13C(6)15N(1)@I': {'composition': 'C(-6)13C(6)N(-1)15N(1)'},
    'Label:13C(9)15N(1)@F': {'composition': 'C(-9)13C(9)N(-1)15N(1)'},
    'Label:13C(6)15N(2)@K': {'composition': 'C(-6)13C(6)N(-2)15N(2)'},
    'Label:13C(6)15N(4)@R': {'composition': 'C(-6)13C(6)N(-4)15N(4)'},
}

FRAGMENT_TYPES = {
    'cid': ['b_z1', 'b_z2', 'y_z1', 'y_z2'],
    'ead': ['c_z1', 'c_z2', 'c_z3', 'z_z1', 'z_z2', 'z_z3'],
    'ead_extended': ['a_z1', 'a_z2', 'b_z1', 'b_z2', 'c_z1', 'c_z2',
                     'c_lossH_z1', 'c_lossH_z2', 'y_z1', 'y_z2',
                     'z_z1', 'z_z2', 'z_addH_z1', 'z_addH_z2'],
}

EXTRA_FRAGMENT_TYPES = {
    'z_lossH': {'ref_ion': 'y', 'delta_formula': 'N(-1)H(-3)',
                'series': 'z', 'loss': 'lossH', 'direction': 'reverse'},
}

NTERM_SERIES = ('a', 'b', 'c')


def register_extra_fragment_types(extra=None, *, verbose=False):
    """Register :data:`EXTRA_FRAGMENT_TYPES` with alphabase."""
    from alphabase.peptide.fragment import (DIRECTION_MAPPING,
                                            FRAGMENT_TYPES as _AB_TYPES,
                                            FragmentType, LOSS_MAPPING,
                                            SERIES_MAPPING, add_new_frag_type)

    types = dict(EXTRA_FRAGMENT_TYPES)
    if extra:
        types.update(extra)
    added = []
    for name, spec in types.items():
        if name in _AB_TYPES:
            continue
        add_new_frag_type(name, f'{spec["ref_ion"]}+{spec["delta_formula"]}')
        _AB_TYPES[name] = FragmentType(
            name=name,
            ref_ion=spec['ref_ion'],
            delta_formula=spec['delta_formula'],
            modloss=False,
            series_id=SERIES_MAPPING[spec['series']],
            loss_id=LOSS_MAPPING[spec['loss']],
            direction_id=DIRECTION_MAPPING[spec['direction']],
        )
        added.append(name)
    if verbose and added:
        print(f'  registered fragment types: {", ".join(added)}')
    return added


def _alphabase_frag_types():
    from alphabase.peptide.fragment import FRAGMENT_TYPES as _AB_TYPES
    return set(_AB_TYPES)


def split_charged_frag_type(name):
    """``'z_addH_z1'`` -> ``('z_addH', 1)``."""
    series, charge = name.rsplit('_z', 1)
    return series, int(charge)


def register_sil_modifications(extra=None, *, verbose=False):
    """Register the stable-isotope labels in :data:`SIL_MODIFICATIONS`."""
    from alphabase.constants.modification import add_new_modifications

    mods = dict(SIL_MODIFICATIONS)
    if extra:
        mods.update(extra)
    add_new_modifications(mods)
    if verbose:
        print(f'  registered {len(mods)} stable-isotope modifications')
    return mods


def target_fragments(targets, *, activation='cid', frag_types=None,
                     min_mz=100.0, max_mz=2000.0):
    """Fragment m/z for each target peptide, as one long table."""
    from alphabase.peptide.fragment import (create_fragment_mz_dataframe,
                                            get_charged_frag_types)
    from alphabase.peptide.precursor import update_precursor_mz

    if frag_types is None:
        if activation not in FRAGMENT_TYPES:
            raise ValueError(f'activation must be one of '
                             f'{sorted(FRAGMENT_TYPES)}, got {activation!r}')
        frag_types = FRAGMENT_TYPES[activation]

    prec = targets.copy().reset_index(drop=True)
    prec['target_idx'] = np.arange(len(prec))
    for col in ('mods', 'mod_sites'):
        if col not in prec.columns:
            prec[col] = ''
        prec[col] = prec[col].fillna('').astype(str)
    prec['nAA'] = prec['sequence'].str.len()
    update_precursor_mz(prec)
    if 'q1_mz' in prec.columns:
        prec['precursor_mz'] = prec['q1_mz'].astype(float)

    parsed = [split_charged_frag_type(t) for t in frag_types]
    if any(s not in _alphabase_frag_types() for s, _ in parsed):
        register_extra_fragment_types()
    unknown = sorted({s for s, _ in parsed if s not in _alphabase_frag_types()})
    if unknown:
        raise ValueError(
            f'unknown fragment series {unknown}; alphabase knows '
            f'{sorted(_alphabase_frag_types())} and this package additionally '
            f'registers {sorted(EXTRA_FRAGMENT_TYPES)}')

    types = get_charged_frag_types(sorted({s for s, _ in parsed}),
                                   max(c for _, c in parsed))
    types = [t for t in types if t in set(frag_types)]
    missing = sorted(set(frag_types) - set(types))
    if missing:
        raise ValueError(f'these fragment types could not be built: {missing}')
    frag_mz = create_fragment_mz_dataframe(prec, types)

    rows = []
    for _, row in prec.iterrows():
        start, stop = int(row['frag_start_idx']), int(row['frag_stop_idx'])
        block = frag_mz.iloc[start:stop]
        n_aa = int(row['nAA'])
        for col in block.columns:
            ion, fragment_charge = split_charged_frag_type(col)
            values = block[col].to_numpy()
            for i, mz in enumerate(values):
                if not np.isfinite(mz) or mz <= 0:
                    continue
                if not (min_mz <= mz <= max_mz):
                    continue
                number = (i + 1 if ion.split('_')[0] in NTERM_SERIES
                          else n_aa - 1 - i)
                rows.append({
                    'target_idx': int(row['target_idx']),
                    'sequence': row['sequence'],
                    'charge': int(row['charge']),
                    'precursor_mz': float(row['precursor_mz']),
                    'ion_type': ion,
                    'ion_number': int(number),
                    'fragment_charge': fragment_charge,
                    'product_mz': float(mz),
                })
    return pd.DataFrame(rows)


def y_ions_above_precursor(fragments, *, ion_types=('y', 'z'), margin=0.0):
    """Keep C-terminal fragments heavier than the precursor."""
    keep = (fragments['ion_type'].isin(ion_types)
            & (fragments['product_mz'] > fragments['precursor_mz'] + margin))
    return fragments[keep]


def top_n_fragments(fragments, intensities, n=6):
    """Keep the `n` most intense fragments of each target."""
    work = fragments.copy()
    work['_intensity'] = np.asarray(intensities, dtype=float)
    out = (work.sort_values('_intensity', ascending=False)
           .groupby('target_idx', sort=False).head(n)
           .drop(columns='_intensity'))
    return out.sort_index()


def summed_fragment_xic(ms_data, fragments, *, rt_range=None, mz_tol='auto',
                        mz_tol_unit='ppm', match='auto', match_tol=None,
                        recentre=True, recentre_rt_range=None,
                        mz_shift_ppm=None, window='central'):
    """One chromatogram per target: its fragments' XICs summed."""
    if len(fragments) == 0:
        return np.array([]), np.array([])
    from .targeted_scans import DEFAULT_MATCH_TH, is_targeted, target_scans
    precursor_mz = float(fragments['precursor_mz'].iloc[0])
    mz_tol = resolve_mz_tol(mz_tol, mz_tol_unit, ms_data, ms_level=2)
    if match == 'auto':
        match = 'precursor' if is_targeted(ms_data) else 'window'
        if match == 'precursor':
            warnings.warn(
                'targeted acquisition detected: selecting scans by precursor '
                'MASS rather than by isolation window. Pass match= to '
                'silence this.', stacklevel=2)
    if match == 'precursor':
        tol = DEFAULT_MATCH_TH if match_tol is None else match_tol
        try:
            view = target_scans(ms_data, precursor_mz, tol=tol, ms_level=2,
                                rt_range=rt_range)
        except ValueError:
            return np.array([]), np.array([])
        xic = extract_xic(view, fragments['product_mz'].to_numpy(),
                          mz_tol=mz_tol, mz_tol_unit=mz_tol_unit, ms_level=2,
                          recentre=recentre,
                          recentre_rt_range=recentre_rt_range,
                          mz_shift_ppm=mz_shift_ppm)
    elif match == 'window':
        xic = extract_xic(
            ms_data, fragments['product_mz'].to_numpy(),
            mz_tol=mz_tol, mz_tol_unit=mz_tol_unit, rt_range=rt_range,
            ms_level=2, precursor_mz=precursor_mz, recentre=recentre,
            recentre_rt_range=recentre_rt_range,
            mz_shift_ppm=mz_shift_ppm, window=window)
    else:
        raise ValueError("match must be 'auto', 'window' or 'precursor', "
                         f'got {match!r}')
    return xic['rt'].to_numpy(), xic['intensity'].to_numpy()


ISOTOPE_SPACING_DA = 1.0033548


def precursor_isotope_transitions(targets, *, n_isotopes=3):
    """The MS1 counterpart of :func:`target_fragments`: one row per isotope."""
    from ..targeted_method import precursor_q1_mz

    if n_isotopes < 1:
        raise ValueError('n_isotopes must be at least 1')
    mono = precursor_q1_mz(targets, isotope='monoisotope')['q1_mz']
    rows = []
    for idx, row in targets.iterrows():
        z = int(row['charge'])
        for k in range(n_isotopes):
            rows.append({**row.to_dict(), 'target_idx': idx,
                         'precursor_mz': float(mono.loc[idx]),
                         'ion_type': 'M', 'ion_number': k,
                         'fragment_charge': z,
                         'product_mz': float(mono.loc[idx])
                         + k * ISOTOPE_SPACING_DA / z})
    return pd.DataFrame(rows)


def quantify_targets(ms_files, fragments, *, rt_range=None, mz_tol='auto',
                     mz_tol_unit='ppm', fragment_filter=y_ions_above_precursor,
                     reference_by='sequence', reference_selector='max_signal',
                     integrate_fwhm=1.5, search_fwhm=1.5, points_fwhm=1.27,
                     max_rt_delta_fwhm=1.0, min_sn=3.0, interpolate_points=500,
                     recentre=True, recentre_fwhm=2.0,
                     min_shift_frac=0.25,
                     baseline='offpeak', decoy_offset_da=6.7,
                     match='auto', match_tol=None, ms_level=2,
                     return_traces=False, verbose=True):
    """Integrate every target in every run, on an anchored window."""
    if ms_level not in (1, 2):
        raise ValueError(f'ms_level must be 1 or 2, got {ms_level!r}')
    if ms_level == 1:
        fragment_filter = None
    frag_by_target = dict(tuple(fragments.groupby('target_idx', sort=True)))
    meta = fragments.drop_duplicates('target_idx').set_index('target_idx')
    groups = (meta.groupby(reference_by).groups if reference_by
              else {t: [t] for t in meta.index})

    rows, all_traces = [], {}
    for run, ms_data in ms_files.items():
        kept = {}
        for target_idx, frags in frag_by_target.items():
            kept[target_idx] = (fragment_filter(frags)
                                if fragment_filter is not None else frags)

        resolved_tol = (resolve_mz_tol(mz_tol, mz_tol_unit, ms_data,
                                       ms_level=ms_level)
                        if mz_tol_unit == 'ppm' else None)

        def _xic(frags, **kw):
            if ms_level == 1:
                x = extract_xic(
                    ms_data, frags['product_mz'].to_numpy(),
                    mz_tol=(resolved_tol if resolved_tol is not None
                            else mz_tol),
                    mz_tol_unit=mz_tol_unit, rt_range=rt_range, ms_level=1,
                    recentre=kw.get('recentre', False),
                    mz_shift_ppm=kw.get('mz_shift_ppm'))
                return x['rt'].to_numpy(), x['intensity'].to_numpy()
            return summed_fragment_xic(
                ms_data, frags, rt_range=rt_range, mz_tol=mz_tol,
                mz_tol_unit=mz_tol_unit, match=match, match_tol=match_tol,
                **kw)

        traces = {t: _xic(f, recentre=False) for t, f in kept.items()}

        anchors = {}
        for key, members in groups.items():
            members = list(members)
            ref = _pick_reference(reference_selector, members, traces, meta,
                                  key)
            rt, inten = traces[ref]
            peak = pick_peak(rt, inten, baseline='p5', min_sn=min_sn,
                             interpolate_points=interpolate_points)
            anchors[key] = (ref, peak)

        decoy_traces = {}
        for key, members in groups.items():
            ref, anchor = anchors[key]
            apex, fwhm = anchor['apex_rt'], anchor['fwhm']
            loc = ((apex - recentre_fwhm * fwhm, apex + recentre_fwhm * fwhm)
                   if np.isfinite(apex) and np.isfinite(fwhm) and fwhm > 0
                   else None)
            shift = np.nan
            if loc is not None and recentre:
                shift = measure_mz_shift_ppm(
                    ms_data, kept[ref]['product_mz'].to_numpy(),
                    fwhm_ppm=(resolved_tol / XIC_FWHM_FRACTION
                              if resolved_tol is not None else None),
                    rt_range=loc, ms_level=ms_level)
            applied = (np.isfinite(shift) and resolved_tol
                       and abs(shift) >= min_shift_frac * resolved_tol)
            for target_idx in members:
                frags = kept[target_idx]
                if applied:
                    traces[target_idx] = _xic(frags, recentre=False,
                                              mz_shift_ppm=shift)
                if decoy_offset_da:
                    decoy = frags.copy()
                    decoy['product_mz'] = decoy['product_mz'] + decoy_offset_da
                    decoy_traces[target_idx] = _xic(
                        decoy, recentre=False,
                        mz_shift_ppm=shift if applied else None)

        for key, members in groups.items():
            ref, anchor = anchors[key]
            for target_idx in members:
                rt, inten = traces[target_idx]
                info = meta.loc[target_idx]
                row = {'run': run, 'ms_level': int(ms_level),
                       'target_idx': int(target_idx),
                       'sequence': info['sequence'], 'charge': int(info['charge']),
                       'precursor_mz': float(info['precursor_mz']),
                       'reference_target_idx': int(ref),
                       'anchor_rt': anchor['apex_rt'],
                       'fwhm': anchor['fwhm'],
                       'mz_shift_ppm': shift,
                       'mz_shift_applied': bool(applied)}
                if rt.size == 0 or not np.isfinite(anchor['apex_rt']):
                    rows.append({**row, 'apex_rt': np.nan, 'rt_delta_s': np.nan,
                                 'trace_r': np.nan,
                                 'area': 0.0, 'sn': np.nan, 'n_points': 0,
                                 'points_across_peak': 0, 'rt_outlier': False,
                                 'decoy_area': np.nan,
                                 'signal_to_decoy': np.nan})
                    continue

                half = search_fwhm * anchor['fwhm']
                local = pick_peak(rt, inten,
                                  rt_range=(anchor['apex_rt'] - half,
                                            anchor['apex_rt'] + half),
                                  baseline='p5',
                                  interpolate_points=interpolate_points)
                apex = (local['apex_rt'] if np.isfinite(local['apex_rt'])
                        else anchor['apex_rt'])
                width = integrate_fwhm * anchor['fwhm']
                integ = integrate_peak(rt, inten, apex - width, apex + width,
                                       baseline=baseline)

                decoy_area = np.nan
                if decoy_offset_da:
                    d_rt, d_int = decoy_traces[target_idx]
                    if d_rt.size:
                        decoy_area = integrate_peak(
                            d_rt, d_int, apex - width, apex + width,
                            baseline=baseline)['area']
                    else:
                        decoy_area = 0.0

                rows.append({
                    **row,
                    'apex_rt': float(apex),
                    'rt_delta_s': float((apex - anchor['apex_rt']) * 60.0),
                    'trace_r': _anchor_correlation(
                        rt, inten, traces[ref], apex, width),
                    'area': integ['area'],
                    'peak_baseline': integ['baseline'],
                    'sn': local['sn'],
                    'n_points': integ['n_points'],
                    'points_across_peak': points_in_window(
                        rt, apex, points_fwhm * anchor['fwhm']),
                    'rt_outlier': bool(
                        anchor['fwhm'] > 0
                        and abs(apex - anchor['apex_rt'])
                        > max_rt_delta_fwhm * anchor['fwhm']),
                    'decoy_area': decoy_area,
                    'signal_to_decoy': (integ['area'] / decoy_area
                                        if decoy_area and decoy_area > 0
                                        else np.inf),
                })
        if return_traces:
            all_traces.update({(run, t): tr for t, tr in traces.items()})
        if verbose:
            print(f'  quantify_targets: {run} done '
                  f'({len(frag_by_target)} targets)', flush=True)
    if return_traces:
        return pd.DataFrame(rows), all_traces
    return pd.DataFrame(rows)


def _pick_reference(selector, members, traces, meta, key):
    """Which target of a group anchors it."""
    if callable(selector):
        return int(selector(members, traces, meta, key))
    if isinstance(selector, dict):
        hits = [t for t in members
                if all(t in meta.index and col in meta.columns
                       and meta.loc[t, col] == val
                       for col, val in selector.items())]
        if len(hits) != 1:
            raise ValueError(
                f'reference_selector {selector!r} matched {len(hits)} targets '
                f'in group {key!r} (needs exactly one); members {list(members)}')
        return int(hits[0])
    if selector == 'max_signal':
        return max(members, key=lambda t: (traces[t][1].max()
                                           if traces[t][1].size else -1))
    return int(selector)


def _anchor_correlation(rt, inten, anchor_trace, apex, half_width):
    """Pearson r of a target's trace against its anchor's, over the peak."""
    a_rt, a_int = anchor_trace
    if rt.size == 0 or a_rt.size == 0 or not np.isfinite(apex):
        return np.nan
    m = np.abs(rt - apex) <= half_width
    if m.sum() <= 3:
        return np.nan
    y = inten[m]
    template = np.interp(rt[m], a_rt, a_int)
    if np.std(y) == 0 or np.std(template) == 0:
        return np.nan
    return float(np.corrcoef(y, template)[0, 1])


def calibration_fit(concentration, area, *, weighting='1/x2',
                    sigma='intercept_se', decoy_area=None,
                    decoy_multiple=3.0):
    """Least-squares calibration line with LOD/LOQ and per-point accuracy."""
    x = np.asarray(concentration, dtype=float)
    y = np.asarray(area, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        raise ValueError('need at least two finite (concentration, area) pairs')

    if weighting == '1/x2':
        w = 1.0 / np.square(x)
    elif weighting == 'none':
        w = np.ones_like(x)
    else:
        raise ValueError(f"weighting must be '1/x2' or 'none', got {weighting!r}")

    sw = w.sum()
    sx, sy = (w * x).sum(), (w * y).sum()
    sxx, sxy = (w * x * x).sum(), (w * x * y).sum()
    denom = sw * sxx - sx * sx
    if abs(denom) <= 1e-12 * max(sw * sxx, sx * sx):
        raise ValueError('calibration is degenerate — all concentrations equal?')
    slope = (sw * sxy - sx * sy) / denom
    intercept = (sxx * sy - sx * sxy) / denom

    pred = slope * x + intercept
    ss_res = (w * np.square(y - pred)).sum()
    ss_tot = (w * np.square(y - (w * y).sum() / sw)).sum()
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    levels = (pd.DataFrame({'concentration': x, 'area': y})
              .groupby('concentration')['area']
              .agg(['mean', 'std', 'count']).reset_index())
    levels['cv_pct'] = levels['std'] / levels['mean'] * 100.0
    levels['measured_concentration'] = (levels['mean'] - intercept) / slope
    levels['accuracy_pct'] = (levels['measured_concentration']
                              / levels['concentration'] * 100.0)

    lowest = levels.loc[levels['concentration'].idxmin()]
    sd_low = float(lowest['std']) if np.isfinite(lowest['std']) else np.nan

    dof = x.size - 2
    if dof >= 1:
        mx = sx / sw
        sxx_c = (w * np.square(x - mx)).sum()
        s2 = (w * np.square(y - pred)).sum() / dof
        intercept_se = float(np.sqrt(s2 * (1.0 / sw + mx * mx / sxx_c)))
    else:
        intercept_se = np.nan

    if sigma == 'intercept_se':
        sigma_blank = intercept_se
    elif sigma == 'lowest_level':
        sigma_blank = sd_low
    else:
        raise ValueError("sigma must be 'intercept_se' or 'lowest_level', "
                         f'got {sigma!r}')
    lod = 3.3 * sigma_blank / slope if slope > 0 else np.nan
    loq = 10.0 * sigma_blank / slope if slope > 0 else np.nan

    decoy_median, lod_decoy = np.nan, np.nan
    if decoy_area is not None:
        d = np.asarray(decoy_area, dtype=float)
        d = d[np.isfinite(d)]
        if d.size:
            decoy_median = float(np.median(d))
            if slope > 0 and decoy_median > 0:
                candidate = (decoy_multiple * decoy_median - intercept) / slope
                lod_decoy = float(candidate) if candidate > 0 else np.nan

    return {'slope': float(slope), 'intercept': float(intercept),
            'r_squared': r_squared, 'weighting': weighting,
            'lod': float(lod), 'loq': float(loq),
            'sigma': sigma, 'sigma_blank': float(sigma_blank),
            'intercept_se': intercept_se, 'sd_lowest_level': sd_low,
            'lod_decoy': lod_decoy, 'decoy_area_median': decoy_median,
            'decoy_multiple': float(decoy_multiple),
            'n_levels': int(len(levels)), 'levels': levels}
