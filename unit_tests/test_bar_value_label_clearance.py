"""A bar's value label must not land on its own replicate points."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import zenotof8600_analytics as core
from zenotof8600_analytics.plotting._style import (BAR_VALUE_PAD_PT,
                                                  BAR_VALUE_POINT_PAD_PT)


def _df():
    return pd.DataFrame({
        'cond': ['a'] * 4 + ['b'] * 4,
        'val': [10.0, 10.1, 9.9, 10.05, 20.0, 21.0, 19.5, 20.5]})


def _label_y_px(ax):
    ax.figure.canvas.draw()
    out = []
    for t in ax.texts:
        bb = t.get_window_extent(ax.figure.canvas.get_renderer())
        out.append((t.get_text(), bb.y0))
    return out


def test_pad_is_larger_when_points_are_drawn():
    assert BAR_VALUE_POINT_PAD_PT > BAR_VALUE_PAD_PT


def test_label_sits_above_the_highest_replicate():
    d = _df()
    fig, ax = plt.subplots()
    core.plot_bar(d, x_col='cond', y_col='val', ax=ax,
                  show_points=True, show_values=True, value_format='float')
    ax.figure.canvas.draw()
    for text, y0 in _label_y_px(ax):
        cond = 'a' if float(text) < 15 else 'b'
        highest = d[d['cond'] == cond]['val'].max()
        y_px = ax.transData.transform((0, highest))[1]
        assert y0 > y_px, (text, cond, y0, y_px)
    plt.close(fig)


def test_offset_is_in_points_not_data_units():
    gaps = []
    for scale in (1.0, 1000.0):
        d = _df()
        d['val'] = d['val'] * scale
        fig, ax = plt.subplots()
        core.plot_bar(d, x_col='cond', y_col='val', ax=ax,
                      show_points=True, show_values=True)
        ax.figure.canvas.draw()
        top = d[d['cond'] == 'b']['val'].max()
        y_px = ax.transData.transform((0, top))[1]
        y0 = max(y for _, y in _label_y_px(ax))
        gaps.append(y0 - y_px)
        plt.close(fig)
    assert gaps[0] == pytest.approx(gaps[1], rel=0.05), gaps


def test_no_points_keeps_the_label_tight_to_the_bar():
    d = _df()
    fig, ax = plt.subplots()
    core.plot_bar(d, x_col='cond', y_col='val', ax=ax,
                  show_points=False, show_values=True)
    ax.figure.canvas.draw()
    mean_b = d[d['cond'] == 'b']['val'].mean()
    y_px = ax.transData.transform((0, mean_b))[1]
    y0 = max(y for _, y in _label_y_px(ax))
    dpi = fig.dpi
    assert (y0 - y_px) < BAR_VALUE_POINT_PAD_PT / 72.0 * dpi
    plt.close(fig)
