"""Fit the provisional Q1 axes (A5, A9) the whole-file Q1 measurement is centred on."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)

import os
import shutil
import subprocess
import sys

import zt8600_config as cfg

EXP = os.path.join(cfg.REPO_ROOT, 'experiments', 'H032_E376')
RUN = '20260905_Zeno2_Eno0_32p8min_TiHe_SA_H032_E376_'
SEARCH = {'A5': '40SPD_1Th_1p48s', 'A9': '40SPD_2p8Th_1p48s'}
DUMP_RT = '12.5,15.5,18.5,21.5,23.5,24.5,27.5,29.5'
CACHE_AXES = cfg.cache('H032_E376', 'q1_calibration')
PACKAGE_AXES = os.path.join(cfg.data('H032_E376'), 'q1_calibration_old')


def run(script, *args):
    print(f'\n>> {script} {" ".join(args)}', flush=True)
    subprocess.run([sys.executable, script, *args], cwd=EXP, check=True)


def fit(tag):
    hdf = cfg.require(cfg.raw('ztscan_full', f'{RUN}{tag}.hdf'), 'ZT Scan HDF')
    features = cfg.require(os.path.join(cfg.data('H032_E376'), SEARCH[tag],
                                        'lfq.dia.features.csv'))
    target = os.path.join(CACHE_AXES, f'q1_axis_{tag}.json')
    if os.path.exists(target):
        raise SystemExit(f'{target} exists -- tag it deleteme_ first; the '
                         f'patch dump reads it and would no longer fit '
                         f'from scratch')
    run('precursor_mass_q1_vs_ms1.py', hdf, features, RUN + tag,
        f'--label={tag}', f'--stem=precursor_mass_q1_vs_ms1_{tag}')
    run('dump_q1_patches.py', hdf, features, RUN + tag, f'--tag={tag}',
        f'--dump={DUMP_RT}')
    run('q1_axis_stability.py', tag)
    os.makedirs(PACKAGE_AXES, exist_ok=True)
    shutil.copy2(target, os.path.join(PACKAGE_AXES, f'q1_axis_{tag}.json'))
    print(f'{tag}: wrote {target} and its data-package copy', flush=True)


def main(tags):
    for tag in tags or ['A5', 'A9']:
        fit(tag)


if __name__ == '__main__':
    main(sys.argv[1:])
