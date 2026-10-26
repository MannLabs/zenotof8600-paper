"""Measure the Promega 6x5 standard on the MRM-HR A-runs of H032_E396."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(cfg.DATA_ROOT, 'H032_E396')
E396_DIR = cfg.data('H032_E396')

RAW_CACHE = cfg.raw('ms_data_cache')

RUNS = ('A1', 'A2', 'A3', 'A4')
PREFIX = '20260921_Zeno2_Eno0_6p4min_TiHe_SA_H032_E396_'


def main():
    """Measure, accept, fit and back-calculate -- all under `rules_6x5`."""
    os.makedirs(DATA, exist_ok=True)
    sys.path.insert(0, E396_DIR)
    import analyze_6x5 as a
    import rules_6x5 as rules
    rules.configure(a)
    print(f'measuring {len(RUNS)} MRM-HR runs: {", ".join(RUNS)}')
    print('--- pass 1: single transitions, detection and ion ratios ---')
    rules.run_pass1(a)
    print('--- pass 2: final transition sets, fit ---')
    a.main()

    import plot_accuracy as pa
    pa.a = a
    pa.SUB = ''
    print('--- back-calculated accuracy ---')
    pa.main()

    for name in ('promega_6x5_targets.csv', 'promega_6x5_masses.csv'):
        src = os.path.join(DATA, 'targets', name)
        dst = os.path.join(DATA, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f'  copied {name}')

if __name__ == '__main__':
    main()
