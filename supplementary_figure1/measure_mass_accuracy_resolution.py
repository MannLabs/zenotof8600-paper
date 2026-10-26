"""Supplementary Figure 1g, h input: per-peak MS1/MS2 mass error and resolution."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import glob
import os
import shutil
import sys

ARMS = {
    '8600': ('H032_E352', '8600_60SPD_100vw_12ms'),
    '7600': ('H032_E303', '7600_60SPD_100vw_12ms'),
}
ALL_PRECURSORS = 10 ** 7
SEARCH_TOL_PPM = 20.0
CALIB_TARGETS = 5000


def main(arm='8600'):
    experiment, search = ARMS[arm]
    import raw_ms_diagnostics as rmd
    out = cfg.output_dir('supplementary_figure1', 'raw_ms_diagnostics', arm)
    rmd._out_dir = lambda: out
    rmd.EXTRA_RAW_DIRS.append(cfg.raw(experiment, 'raw'))
    rmd.CALIB_TOL_PPM = SEARCH_TOL_PPM
    rmd.CALIB_INTENSITY_FRAC = 1.0
    rmd.CALIB_MAX_TARGETS = CALIB_TARGETS
    cache = cfg.raw('ms_data_cache')
    rmd.configure(data_root=cfg.data(experiment), use_raw=True, force=True,
                  n_precursors=ALL_PRECURSORS, top_intensity_frac=1.0,
                  cache_dir=cache,
                  conditions={search: cfg.data(experiment, search,
                                               'lfq.dia.features.csv')})
    rmd.main()
    found = glob.glob(os.path.join(cache, f'*{experiment}*.rawdiag.parquet'))
    if not found:
        raise SystemExit('no *.rawdiag.parquet produced')
    for f in found:
        shutil.copy2(f, cfg.data(experiment, os.path.basename(f)))
        print(f'{f} -> {cfg.data(experiment)}')


if __name__ == '__main__':
    main(*sys.argv[1:2])
