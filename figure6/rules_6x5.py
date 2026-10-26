"""The paper's rules for the Promega 6x5 standard, applied to E396's driver."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os

import numpy as np
import pandas as pd

import zenotof8600_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(cfg.DATA_ROOT, 'H032_E396')
RAW_CACHE = cfg.raw('ms_data_cache')
RUNS = ('A1', 'A2', 'A3', 'A4')
PREFIX = '20260921_Zeno2_Eno0_6p4min_TiHe_SA_H032_E396_'

N_TRANSITIONS = 6
MIN_RESIDUES = 4
MIN_GOOD_TRANSITIONS = 3
EC_TOLERANCE = ((50, 20), (20, 25), (10, 30), (0, 50))
MIN_REL_INTENSITY = 5.0
MIN_FRACTION_WITHIN = 2 / 3
IDENTITY = 'trace'
MIN_TRACE_R = 0.80

FINAL = {}
ACCEPTED = {}
SAVED = os.path.join(DATA, '6x5_rules_result.json')


def save():
    import json
    with open(SAVED, 'w') as fh:
        json.dump({'transitions': {p: [list(k) for k in v]
                                   for p, v in FINAL.items()},
                   'accepted': [[p, lv, ok] for (p, lv), ok in ACCEPTED.items()]},
                  fh, indent=1)


def load():
    import json
    with open(SAVED) as fh:
        r = json.load(fh)
    FINAL.clear()
    ACCEPTED.clear()
    FINAL.update({p: [tuple(k) for k in v] for p, v in r['transitions'].items()})
    ACCEPTED.update({(p, int(lv)): bool(ok) for p, lv, ok in r['accepted']})


def configure(a, out_dir=None, use_saved=False):
    """Re-point E396's driver at this figure's data and install the rules."""
    if use_saved:
        load()
    a.HERE = a.pathlib.Path(DATA)
    a.RAW_DIR = a.pathlib.Path(cfg.raw('H032_E396', 'raw'))
    a.CACHE_DIR = RAW_CACHE
    a.PREFIX = PREFIX
    a.RUNS = RUNS
    a.REFERENCE_RUN = RUNS[0]
    a.OUT_DIR = a.pathlib.Path(out_dir or DATA)
    a.SELECT_MIN_R_BY_PEPTIDE = {}
    a.SELECT_MIN_SN_BY_PEPTIDE = {}
    a.MIN_TRACE_R_BY_PEPTIDE = {}
    a.choose_transitions = lambda tg, frags: choose_transitions(a, tg, frags)
    a.accepted_levels = accepted_levels
    return a


def choose_transitions(a, tg, frags):
    """{peptide: [(ion_type, number, charge)]}; FINAL sets once pass 1 ran."""
    picked, cand = {}, []
    for pep, grp in tg.groupby('sequence'):
        if pep in FINAL:
            picked[pep] = FINAL[pep]
            continue
        ref = grp[grp['level'] == a.REFERENCE_LEVEL].iloc[0]
        cand.append(frags[(frags['target_idx'] == ref.name)
                          & (frags['ion_type'] == 'y')
                          & (frags['fragment_charge'] == 1)
                          & (frags['ion_number'] >= MIN_RESIDUES)
                          & (frags['ion_number'] <= len(pep) - 1)]
                    .assign(sequence=pep))
    if not cand:
        return picked
    c = pd.concat(cand).copy()
    c['key'] = list(zip(c['ion_type'], c['ion_number'].astype(int),
                        c['fragment_charge'].astype(int)))
    c['target_idx'] = np.arange(len(c))
    q = core.raw.quantify_targets(
        {a.REFERENCE_RUN: a._load(a.REFERENCE_RUN)}, c,
        reference_by='sequence', fragment_filter=None, match='precursor',
        match_tol=a.Q1_TOL, ms_level=2, verbose=False)
    c = c.join(q.set_index('target_idx')['area'], on='target_idx')
    for pep, g in c.groupby('sequence'):
        g = g.sort_values('area', ascending=False)
        g = g[g['area'] >= MIN_REL_INTENSITY / 100 * g['area'].max()]
        sel = list(g['key'].head(N_TRANSITIONS))
        if len(sel) < MIN_GOOD_TRANSITIONS:
            raise SystemExit(f'{pep}: only {len(sel)} candidate y ions')
        picked[pep] = sel
        print(f'  {pep:12s} ' + ', '.join(f'{t}{n}+' for t, n, _z in sel)
              + f'   [brightest y1+ >= y{MIN_RESIDUES} by integrated area, '
                f'level {a.REFERENCE_LEVEL}, {a.REFERENCE_RUN}]')
    return picked


def ec_tolerance(rel_pct):
    for lo, tol in EC_TOLERANCE:
        if rel_pct > lo:
            return tol
    return EC_TOLERANCE[-1][1]


def run_pass1(a):
    """Single-transition and summed measurement -> FINAL sets and ACCEPTED."""
    FINAL.clear()
    ACCEPTED.clear()
    tg = a.method_table()
    sel = a._ms2_transitions(tg).copy()
    runs = {r: a._load(r) for r in a.RUNS}
    kw = dict(reference_by='sequence', fragment_filter=None, match='precursor',
              match_tol=a.Q1_TOL, ms_level=2, verbose=False)
    summed = core.raw.quantify_targets(runs, sel, **kw)
    summed = summed.join(tg[['sequence', 'level']].rename(
        columns={'sequence': 'peptide'}), on='target_idx')
    one = sel.copy()
    one['orig_target'] = one['target_idx']
    keys = list(zip(one['ion_type'], one['ion_number'].astype(int),
                    one['fragment_charge'].astype(int)))
    one['transition'] = [f'{t}{n}+{z}' for t, n, z in keys]
    keymap = dict(zip(one['transition'], keys))
    one['target_idx'] = np.arange(len(one))
    d = core.raw.quantify_targets(runs, one, **kw)
    d = d.join(one.set_index('target_idx')[['orig_target', 'transition']],
               on='target_idx')
    d = d.join(tg[['sequence', 'level']].rename(
        columns={'sequence': 'peptide'}), on='orig_target')
    d['area'] = d['area'].clip(lower=0)

    rows = []
    for pep, g in d.groupby('peptide'):
        top = g['level'].max()
        mean = g.groupby(['level', 'transition'])['area'].mean().unstack()
        base = mean.loc[top].idxmax()
        rel_top_pct = 100 * mean.loc[top] / mean.loc[top, base]
        to_top = mean.div(mean.loc[top], axis=1)
        current = list(mean.columns)
        accepted = {top: True}
        for lv in sorted(mean.index, reverse=True):
            if lv == top:
                continue
            ref_ratio = float(np.median(to_top.loc[lv, current]))
            with np.errstate(divide='ignore', invalid='ignore'):
                dev = 100 * (to_top.loc[lv] / ref_ratio - 1)
            within = [t for t in current
                      if abs(dev[t]) <= ec_tolerance(rel_top_pct[t])]
            for t in mean.columns:
                rows.append({'peptide': pep, 'level': int(lv),
                             'transition': t,
                             'is_base': t == base,
                             'rel_intensity_top_pct': rel_top_pct[t],
                             'ec_deviation_pct': dev[t],
                             'ec_tolerance_pct': ec_tolerance(rel_top_pct[t]),
                             'in_current_set': t in current})
            need = max(MIN_GOOD_TRANSITIONS,
                       int(np.ceil(MIN_FRACTION_WITHIN * len(current))))
            if IDENTITY == 'trace':
                tr = summed[(summed['peptide'] == pep)
                            & (summed['level'] == lv)]['trace_r'].mean()
                ident = bool(tr >= MIN_TRACE_R)
            else:
                ident = len(within) >= need
            ok = ident
            accepted[lv] = ok
            if not ok:
                break
        for lv in mean.index:
            ACCEPTED[(pep, int(lv))] = bool(accepted.get(lv, False))
        FINAL[pep] = [keymap[t] for t in current]
        print(f'  {pep:12s} final set ' + ', '.join(current)
              + '   accepted levels '
              + ', '.join(str(int(lv)) for lv in sorted(accepted)
                          if accepted[lv]))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.OUT_DIR, '6x5_ion_ratio_check.csv'), index=False)
    save()
    return t


def accepted_from_table(g):
    """{level: bool} from a per-(run, level) table of ONE peptide."""
    ok = g.groupby('level')['trace_r'].mean() >= MIN_TRACE_R
    return ok.sort_index(ascending=False).cummin().sort_index()


def accepted_levels(pep, g):
    """The curve-entry rule installed into E396's driver (see above)."""
    return accepted_from_table(g)
