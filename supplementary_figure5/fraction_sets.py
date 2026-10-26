"""Supplementary figure 5f -- cumulative peptides against fractions, EVERY set."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = cfg.panels(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import figure_style as fs
sys.path.insert(0, os.path.join(ROOT, 'figure5'))
import figure5 as f5

LETTER = 'e'
SETS = (6, 12, 24)
AXES_W_IN = 2.95
SHOT_GREY = '#7A7A7A'


def single_shot_levels(cum, arms):
    """The 200 SPD ZT single shot per species -- figure 5's own rule."""
    out = {}
    for species in cum['species'].unique():
        ref = arms[(arms['species'] == species) & (arms['spd'] == 200)
                   & (arms['method'] == 'ZT')]
        if ref.empty:
            raise ValueError(f'no 200 SPD ZT {species} arm')
        out[species] = {c: float(ref[c].iloc[0])
                        for c in ('protein_mean', 'peptide_mean')}
    return out


def main():
    fs.init()
    fs.start_build(script=__file__, inputs=[f5.DATA_E386, f5.DATA_E392],
                   out_dir=OUT)
    cum = f5.cumulative_depth(f5.load_e392())
    single_shot = single_shot_levels(cum, f5.load_e386())
    key, value_col, ref_col, y_label, stem = next(
        p for p in f5.FRACTION_PANELS if p[0] == 'fractions_pep')
    f5._fraction_panel(cum, single_shot, key=key, value_col=value_col,
                       ref_col=ref_col, y_label=y_label, stem=stem,
                       sets=SETS, out=OUT, axes_w_in=AXES_W_IN,
                       shot_color=SHOT_GREY, legend_colspacing=1.0,
                       name=f'supplementary_figure5{LETTER}_fractionation_'
                            f'peptides_all_sets')
    fs.report_stale(OUT, prefix='supplementary_figure')


if __name__ == '__main__':
    main()
