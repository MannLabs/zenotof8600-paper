"""Fit the A9 (2.8 Th) Q1 axis on the whole-file placements, for the zoom."""

import json
import os

import figure2 as f2
from zenotof8600_analytics.ztscan.calibration import fit_q1_axis

TAG = 'A9'


def main():
    if os.path.exists(f2.Q1_AXIS_WHOLEFILE):
        raise SystemExit(f'{f2.Q1_AXIS_WHOLEFILE} exists -- tag it deleteme_ '
                         f'first; this script never overwrites a calibration')
    d = f2.wholefile_placements(TAG)
    axis = fit_q1_axis(d['q1'], d['mz'], d['rt'],
                       loess_frac=f2.Q1_AXIS_LOESS_FRAC, n_knots=80)
    axis.update(source='wholefile_strict_ge_median', run=TAG,
                loess_frac=f2.Q1_AXIS_LOESS_FRAC, min_pct=f2.Q1_MIN_PCT)
    with open(f2.Q1_AXIS_WHOLEFILE, 'w') as fh:
        json.dump(axis, fh, indent=1)
    print(f'{TAG}: fitted on {len(d):,} precursors, '
          f'{d["mz"].min():.0f}-{d["mz"].max():.0f} Th, RT slope '
          f'{axis["slope_rt"]:+.5f} Th/min; wrote {f2.Q1_AXIS_WHOLEFILE}')


if __name__ == '__main__':
    main()
