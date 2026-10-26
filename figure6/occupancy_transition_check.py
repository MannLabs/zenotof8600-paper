"""Does one quant transition bias the pS129 occupancy?"""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
import sys

import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(cfg.DATA_ROOT, 'H032_E397', 'transition_check')
E397 = cfg.data('H032_E397')

SETS = {
    'all five (figure 6g)': (('y', 3), ('z', 4), ('y', 4), ('z', 5), ('z', 6)),
    'without y3': (('z', 4), ('y', 4), ('z', 5), ('z', 6)),
    'z ions only': (('z', 4), ('z', 5), ('z', 6)),
}


def occupancy(path):
    d = pd.read_csv(path)
    s = (d.groupby(['run', 'panel', 'form', 'base'])['area'].sum()
          .unstack(['form', 'base']).reset_index())
    lp = s[('l', 'pS129')] / s[('h', 'pS129')]
    lu = s[('l', 'unmod')] / s[('h', 'unmod')]
    return pd.DataFrame({'run': s['run'], 'condition': s['panel'],
                         'occupancy': 100 * lp / (lp + lu)})


def main():
    sys.path.insert(0, E397)
    import analyze_E397_lib as lib
    rows = []
    for name, qset in SETS.items():
        out = lib.e395.pathlib.Path(OUT, name.split(' (')[0].replace(' ', '_'))
        out.mkdir(parents=True, exist_ok=True)
        lib.OUT_DIR = out
        lib.e395.QUANT_SET = qset
        print(f'\n##### {name}: {qset}')
        lib.main()
        o = occupancy(out / 'e397_lib_measurements.csv')
        a = o[o.condition == 'PFF-'].occupancy
        b = o[o.condition == 'PFF+'].occupancy
        rows.append({'transitions': name, 'pff_minus_pct': a.mean(),
                     'pff_minus_sd': a.std(), 'pff_plus_pct': b.mean(),
                     'pff_plus_sd': b.std(), 'fold': b.mean() / a.mean(),
                     'welch_p': stats.ttest_ind(b, a, equal_var=False).pvalue})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, 'occupancy_by_transition_set.csv'), index=False)
    print('\n' + t.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
