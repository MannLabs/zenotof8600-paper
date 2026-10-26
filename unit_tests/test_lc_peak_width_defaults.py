"""lc_peak_width_analysis samples the top 50% of precursors by default."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lc_peak_width_analysis as lpw


def test_default_top_fraction_is_half():
    assert lpw.TOP_FRACTION == 0.5


def test_top_fraction_is_a_fraction():
    assert 0 < lpw.TOP_FRACTION <= 1
