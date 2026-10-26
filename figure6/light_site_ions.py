"""Are the site-determining ions visible for the ENDOGENOUS (light) pS129 peptide?"""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
import pathlib
import sys

import numpy as np
import pandas as pd

import zenotof8600_analytics as core

sys.path.insert(0, cfg.data('H032_E395'))
import analyze_E395 as e395

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(cfg.DATA_ROOT, 'H032_E397', 'light_site_ions.csv')
RAW_DIR = pathlib.Path(cfg.raw('H032_E397', 'raw'))
CACHE_DIR = cfg.raw('ms_data_cache')
PREFIX = '20260921_Zeno2_Eno0_6p4min_TiHe_SA_H032_E397_'
RUNS = ['B1', 'B2', 'B3', 'B4_repeat']
SITE_IONS = [(t, n) for t in ('y', 'z') for n in range(8, 14)]


def main():
    sp, frags = e395._species()
    e395.QUANT_SET = tuple(SITE_IONS)
    e395.DIAGNOSTIC = ('y', 3)
    pick = {}
    for i, t in sp.iterrows():
        cid = t['compound_id']
        if 'pS129' in cid and '_ox' not in cid:
            pick['light' if '_l_' in cid else 'heavy'] = (i, cid, t)
    print({k: v[1] for k, v in pick.items()})
    rows = []
    for run in RUNS:
        ms = core.raw.load_ms_data(
            str(RAW_DIR / f'{PREFIX}{run}.wiff'), cache_dir=CACHE_DIR,
            reader_options={'keep_k_peaks_per_spec': 10 ** 7})
        tr = {}
        for form, (i, cid, t) in pick.items():
            sub = frags[(frags['target_idx'] == i)
                        & (frags['fragment_charge'] == 1)]
            tr[form] = e395._traces(ms, float(t['q1_mz']), sub,
                                    e395.RT_SCHEDULED[cid])
        apex = e395._summed_apex(tr['heavy'])
        for form in ('light', 'heavy'):
            for ion, (irt, iy) in tr[form].items():
                ions, base, n, sigma = e395._integrate(irt, iy, apex)
                rows.append({'run': run, 'form': form, 'ion': ion,
                             'ions': ions, 'sigma_ions': sigma})
    d = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    d.to_csv(OUT, index=False)
    s = d.pivot_table(index='ion', columns='form', values='ions',
                      aggfunc='sum')
    s['sigma_light'] = np.sqrt(d[d.form == 'light'].groupby('ion')[
        'sigma_ions'].apply(lambda x: (x ** 2).sum()))
    s['L/H'] = s['light'] / s['heavy']
    order = [f'{t}{n}' for t, n in SITE_IONS]
    print(s.reindex([o for o in order if o in s.index]).round(3).to_string())
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
