"""Replicate-point sizing scales with the drawn bar width, not the bar count."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import zenotof8600_analytics as core
from zenotof8600_analytics.plotting import _style


@pytest.fixture(autouse=True)
def restore_scale():
    before = (_style.POINT_DIAMETER_FRAC, _style.POINT_DIAMETER_MIN_PT,
              _style.POINT_DIAMETER_MAX_PT)
    yield
    core.set_replicate_point_scale(*before)


def test_area_scales_as_the_square_of_the_bar_width():
    wide_clamp = dict(min_pt=0.0, max_pt=1e4)
    narrow = core.replicate_point_size(0.16, **wide_clamp)
    wide = core.replicate_point_size(0.32, **wide_clamp)
    assert narrow == pytest.approx(wide / 4.0)


def test_default_clamp_band_is_where_the_paper_bar_widths_live():
    low = _style.POINT_DIAMETER_MIN_PT / (_style.POINT_DIAMETER_FRAC * 72.0)
    high = _style.POINT_DIAMETER_MAX_PT / (_style.POINT_DIAMETER_FRAC * 72.0)
    assert low < 0.15 < high
    assert low < 0.25 < high


def test_diameter_is_the_configured_fraction_of_the_bar():
    size = core.replicate_point_size(0.20, frac=0.45, min_pt=0, max_pt=100)
    assert np.sqrt(size) == pytest.approx(0.45 * 0.20 * 72.0)


def test_diameter_is_clamped_at_both_ends():
    assert np.sqrt(core.replicate_point_size(0.001)) == pytest.approx(
        _style.POINT_DIAMETER_MIN_PT)
    assert np.sqrt(core.replicate_point_size(10.0)) == pytest.approx(
        _style.POINT_DIAMETER_MAX_PT)


def test_set_replicate_point_scale_takes_effect_and_validates():
    core.set_replicate_point_scale(frac=0.9, min_pt=1.0, max_pt=50.0)
    assert np.sqrt(core.replicate_point_size(0.20)) == pytest.approx(
        0.9 * 0.20 * 72.0)
    with pytest.raises(ValueError):
        core.set_replicate_point_scale(min_pt=10.0, max_pt=1.0)


def test_drawn_bar_width_is_measured_from_the_rendered_geometry():
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.bar([0, 1, 2], [1.0, 2.0, 3.0], width=0.5)
    ax.set_xlim(-0.5, 2.5)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    assert core.drawn_bar_width_inches(ax) == pytest.approx(0.5 / 3 * 4.0,
                                                            rel=1e-3)
    ax.set_position([0.0, 0.0, 0.5, 1.0])
    assert core.drawn_bar_width_inches(ax) == pytest.approx(0.5 / 3 * 2.0,
                                                            rel=1e-3)
    plt.close(fig)


def test_drawn_bar_width_is_none_without_bars():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    assert core.drawn_bar_width_inches(ax) is None
    plt.close(fig)


def test_scale_replicate_points_applies_the_measured_size():
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.bar([0, 1, 2], [1.0, 2.0, 3.0], width=0.5)
    ax.set_xlim(-0.5, 2.5)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    dots = ax.scatter([0, 1, 2], [1.0, 2.0, 3.0], s=1.0)

    applied = core.scale_replicate_points(ax, [dots])
    expected = core.replicate_point_size(core.drawn_bar_width_inches(ax))
    assert applied == pytest.approx(expected)
    assert dots.get_sizes()[0] == pytest.approx(expected)
    plt.close(fig)


def test_scale_replicate_points_returns_none_without_bars():
    fig, ax = plt.subplots()
    dots = ax.scatter([0, 1], [0, 1], s=3.0)
    assert core.scale_replicate_points(ax, [dots]) is None
    assert dots.get_sizes()[0] == pytest.approx(3.0)
    plt.close(fig)


def test_bar_count_fallback_still_works():
    from zenotof8600_analytics.plotting._style import _replicate_point_size
    assert _replicate_point_size(1) == 24
    assert _replicate_point_size(1000) == 5
    assert _replicate_point_size(12) == pytest.approx(15.0)


def test_set_axes_size_inches_pins_the_data_area():
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot([0, 1], [0, 1])
    ax.set_ylabel('a label that widens the left margin')
    fig.tight_layout()
    before = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    left_before = before.x0 * fig_w

    core.set_axes_size_inches(fig, ax, w_in=2.6, h_in=2.5)
    pos = ax.get_position()
    w, h = fig.get_size_inches()
    assert pos.width * w == pytest.approx(2.6, abs=1e-3)
    assert pos.height * h == pytest.approx(2.5, abs=1e-3)
    assert pos.x0 * w == pytest.approx(left_before, abs=1e-3)
    plt.close(fig)


def test_set_axes_size_inches_moves_a_twinned_pair_together():
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax2 = ax.twinx()
    fig.tight_layout()
    core.set_axes_size_inches(fig, [ax, ax2], h_in=2.0)
    assert ax.get_position().bounds == pytest.approx(ax2.get_position().bounds)
    assert ax2.get_position().height * fig.get_size_inches()[1] == pytest.approx(
        2.0, abs=1e-3)
    plt.close(fig)


def test_set_axes_size_inches_leaves_omitted_dimension_alone():
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    fig.tight_layout()
    width_before = ax.get_position().width * fig.get_size_inches()[0]
    core.set_axes_size_inches(fig, ax, h_in=1.5)
    assert ax.get_position().width * fig.get_size_inches()[0] == pytest.approx(
        width_before, abs=1e-3)
    plt.close(fig)
