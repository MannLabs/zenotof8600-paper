"""Figure 1d input: summed ion current per acquisition cycle, MS1 and MS2, per run."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os

import pandas as pd

import zenotof8600_analytics as core

RUNS = {
    '7600+': ('H032_E303', '20260417_Zeno1_Eno01_22p4min_TiHe_SA_H032_E303_',
              ['A1', 'A2', 'A3', 'A4_repeat']),
    '8600': ('H032_E352', '20260805_Zeno2_Eno0_22p4min_TiHe_SA_H032_E352_',
             ['A1', 'A2', 'A3', 'A4']),
}
OUT = cfg.data('figure1', 'ion_current')
CACHE = cfg.raw('ms_data_cache')


def measure(instrument, experiment, prefix, well):
    """Per-cycle ion current of one run at MS1 and MS2."""
    ms = core.raw.load_ms_data(cfg.raw(experiment, 'raw', f'{prefix}{well}.wiff'),
                               cache_dir=CACHE,
                               reader_options={'keep_k_peaks_per_spec': 10 ** 7})
    parts = []
    for level in (1, 2):
        t = core.tic(ms, ms_level=level, per='cycle')
        parts.append(pd.DataFrame({'instrument': instrument, 'run': well,
                                   'ms_level': level, 'rt_min': t['rt'],
                                   'tic': t['tic']}))
    return pd.concat(parts, ignore_index=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    for instrument, (experiment, prefix, wells) in RUNS.items():
        for well in wells:
            t = measure(instrument, experiment, prefix, well)
            name = 'A4_repeat' if well == 'A4_repeat' else well
            t.to_csv(os.path.join(OUT, f'{instrument.rstrip("+")}_{name}.csv'),
                     index=False)
            print(f'{instrument} {well}: {len(t)} rows')


if __name__ == '__main__':
    main()
