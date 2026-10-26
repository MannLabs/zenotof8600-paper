"""Supplementary Figure 5 input: LC peak width of the 500 SPD runs of H032_E350."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import sys

sys.path.insert(0, _os.path.join(cfg.REPO_ROOT, 'workflows'))
import technical_qc_analysis as tqc
import lc_peak_width_analysis as lc


def main():
    tqc.RAW_SEARCH_DIRS = (cfg.raw('supplementary_figure5'),)
    root = cfg.data('H032_E350')
    lc.main(data_root=root, out_dir=root, runs_per_condition='all',
            top_fraction=0.5, cache_dir=cfg.raw('ms_data_cache'))


if __name__ == '__main__':
    main()
