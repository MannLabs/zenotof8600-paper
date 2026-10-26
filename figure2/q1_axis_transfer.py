"""Does a Q1 axis calibration fitted on one file carry over to another?"""

import os
import zlib

import numpy as np
import pandas as pd

import figure2 as f2
import proteomics_analytics as core
from proteomics_analytics.ztscan.axis import q1_correction
from proteomics_analytics.ztscan.calibration import fit_q1_axis

SOURCE, TARGET = 'A9', 'A10'
BINS = 12
OUT = os.path.join(f2.DATA, 'q1_axis_transfer_A9_to_A10.csv')


def fit(d):
    return fit_q1_axis(d['q1'], d['mz'], d['rt'],
                       loess_frac=f2.Q1_AXIS_LOESS_FRAC, n_knots=80)


def score(stage, te, res):
    bb = core.binned_stats(te['mz'], res, bins=BINS)
    return {'stage': stage, 'n_test': len(te),
            'median_th': float(np.median(res)),
            'sigma_iqr_th': float(core.sigma_iqr(res)),
            'binned_median_rms_th': float(np.sqrt(np.mean(bb['median'] ** 2))),
            'binned_median_maxabs_th': float(np.max(np.abs(bb['median'])))}


def main():
    src = f2.wholefile_placements(SOURCE)
    tgt = f2.wholefile_placements(TARGET)
    held = tgt['seq'].map(lambda s: zlib.crc32(s.encode()) % 2 == 1)
    tr, te = tgt[~held], tgt[held]
    dev = te['q1'] - te['mz']
    rows = [score('uncalibrated', te, dev)]
    for stage, axis in [(f'own axis ({TARGET}, other half)', fit(tr)),
                        (f'{SOURCE} axis, transferred', fit(src))]:
        rows.append(score(stage, te, dev - q1_correction(axis, te['q1'], te['rt'])))
    out = pd.DataFrame(rows).assign(n_source=len(src), n_target_train=len(tr))
    print(f'{SOURCE}: {len(src):,} precursors; {TARGET}: train {len(tr):,}, '
          f'test {len(te):,}')
    print(out.drop(columns=['n_source', 'n_target_train']).round(4)
          .to_string(index=False))
    out.to_csv(OUT, index=False)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
