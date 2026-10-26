"""Isolation windows as acquired in H032_E352 A1, read from the raw file."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os

import zenotof8600_analytics as core

RUN = '20260805_Zeno2_Eno0_22p4min_TiHe_SA_H032_E352_A1.wiff'
OUT = os.path.join(cfg.data('supplementary_figure8'), 'acquired_windows_E352_A1.csv')


def windows_from_raw(path):
    ms = core.raw.load_ms_data(path, cache_dir=cfg.raw('ms_data_cache'))
    sd = ms.spectrum_df
    w = (sd[sd['ms_level'] == 2][['isolation_lower_mz', 'isolation_upper_mz']]
         .drop_duplicates().sort_values('isolation_lower_mz'))
    return w.round(4).reset_index(drop=True)


def main():
    w = windows_from_raw(cfg.require(cfg.raw('H032_E352', 'raw', RUN), 'raw file'))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    w.to_csv(OUT, index=False)
    print(f'{len(w)} windows, {w.isolation_lower_mz.min():.2f}-'
          f'{w.isolation_upper_mz.max():.2f} m/z; wrote {OUT}')


if __name__ == '__main__':
    main()
