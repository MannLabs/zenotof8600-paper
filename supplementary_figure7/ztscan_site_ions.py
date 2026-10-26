"""Site-determining CID ions of the ENDOGENOUS pS129 peptide, from ZT Scan DIA."""

import os
import sys

import numpy as np
import pandas as pd

import zenotof8600_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_ps129_ztscan as mz

RUNS = [r for g, rr in mz.GROUPS if g == 'PFF+' for r in rr]
OUT = os.path.join(mz.OUT, 'ztscan_site_ions.csv')
SITE = range(8, 14)


def main():
    core.raw.register_sil_modifications()
    sp_all, fr_all = mz.species_and_fragments()
    fr_all = mz.drop_isobaric(sp_all, fr_all)
    sp = sp_all[sp_all['species'] == 'pS129']
    fr = fr_all[fr_all['target_idx'].isin(sp.index)
                & fr_all['ion_number'].isin(SITE)]
    h = int(sp.index[sp['form'] == 'h'][0])
    l = int(sp.index[sp['form'] == 'l'][0])
    rows = []
    for run in RUNS:
        path = mz.hdf(run)
        tr = mz.channel_traces(path, sp, fr, mz.half_isolation(path))
        rt_h, pc_h, ions_h = tr[h]
        pk = core.raw.pick_peak(rt_h, pc_h.sum(axis=1), baseline='offpeak',
                                interpolate_points=500)
        apex, fwhm = float(pk['apex_rt']), float(pk['fwhm'])
        w = mz.INTEGRATE_FWHM * fwhm
        for t, form in ((h, 'heavy'), (l, 'light')):
            rt, pc, ions = tr[t]
            for k, ion in enumerate(ions):
                a = core.raw.integrate_peak(rt, pc[:, k], apex - w, apex + w,
                                            baseline='offpeak')
                rows.append({'run': run, 'form': form, 'ion': ion,
                             'area': float(a['area'])})
        print(f'  {run}: apex {apex:.3f} min, FWHM {fwhm * 60:.1f} s')
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    s = d.pivot_table(index='ion', columns='form', values='area',
                      aggfunc='sum')
    s['L/H'] = s['light'] / s['heavy']
    s['ions_light'] = s['light'] / 20.0
    print(s.round(3).to_string())
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
