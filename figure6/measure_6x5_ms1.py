"""Measure the Promega 6x5 standard from the MS1 SURVEY."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(cfg.DATA_ROOT, 'H032_E396')
OUT = os.path.join(DATA, 'ms1')
RAW_CACHE = cfg.raw('ms_data_cache')
E396_DIR = cfg.data('H032_E396')

RUNS = ('A1', 'A2', 'A3', 'A4')
PREFIX = '20260921_Zeno2_Eno0_6p4min_TiHe_SA_H032_E396_'


def main():
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, E396_DIR)
    import analyze_6x5 as a

    import rules_6x5
    rules_6x5.configure(a, out_dir=OUT, use_saved=True)
    print(f'MS1 measurement of {len(RUNS)} MRM-HR runs -> {OUT}')
    a.main(ms_level=1)


if __name__ == '__main__':
    main()
