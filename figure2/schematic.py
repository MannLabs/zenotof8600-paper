"""Figure 2, panel a -- Zeno SWATH DIA against ZT Scan, as a schematic."""

import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_r, 'zt8600_config.py')):
    _r = _os.path.dirname(_r)
_sys.path.insert(0, _r)
import zt8600_config as cfg

import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

import zenotof8600_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = cfg.panels(__file__)

sys.path.insert(0, os.path.dirname(HERE))
import figure_style as fs

PULSE_DT = 1.0
N_TIME = 18.0

DWELL = 4
OVERHEAD = 2.0

Q1_LO, Q1_HI = 0.0, 10.0
WINDOW_W = 10.0 / 3.0

ACTIVE = core.PALETTE_SINGLE[2]
OVERHEAD_C = core.PALETTE_SINGLE[0]

PRECURSORS = (core.PALETTE_SINGLE[0], core.PALETTE_SINGLE[2],
              core.PALETTE_SINGLE[3])

PRECURSOR_DM = (-1.05, 0.0, 1.05)

BYSTANDERS = (0.6, 1.5, 2.3, 7.6, 8.4, 9.3)

PRECURSOR_INT = (0.55, 1.0, 0.78)

N_BINS = 13
BIN_GAP = 0.22

PULSE_HALF = 0.45

EDGE = '#1A1A1A'
EDGE_LW = 0.6


def _face(color, alpha):
    """`color` at `alpha`, as an RGBA tuple, so a solid edge can sit on it."""
    return (*mcolors.to_rgb(color), alpha)


def _pulse_times():
    """Zeno SWATH and ZT Scan pulse times on one shared clock."""
    swath, t, k = [], 0.0, 0
    while t < N_TIME:
        for i in range(DWELL):
            if t >= N_TIME:
                break
            swath.append((t, k))
            t += PULSE_DT
        t += OVERHEAD
        k += 1
    scan = list(np.arange(0.0, N_TIME, PULSE_DT))
    return swath, scan


def _time_row(ax, mode, swath, scan):
    """Q1 position against time, with every TOF pulse drawn."""
    ax.set_xlim(-0.5, N_TIME + 0.5)
    ax.set_ylim(Q1_LO, Q1_HI)
    half = PULSE_DT * PULSE_HALF

    if mode == 'swath':
        t, k = 0.0, 0
        while t < N_TIME:
            t += DWELL * PULSE_DT
            if t < N_TIME:
                ax.add_patch(Rectangle((t, Q1_LO), OVERHEAD,
                                       Q1_HI - Q1_LO, facecolor=OVERHEAD_C,
                                       alpha=0.20, edgecolor=OVERHEAD_C,
                                       linestyle='--', linewidth=0.7,
                                       zorder=1))
            t += OVERHEAD
            k += 1
        for t, k in swath:
            y = Q1_LO + WINDOW_W * (k + 0.5)
            if y > Q1_HI:
                continue
            ax.add_patch(Rectangle((t, y - WINDOW_W / 2),
                                   2 * half, WINDOW_W,
                                   facecolor=_face(ACTIVE, 0.55),
                                   edgecolor=EDGE, linewidth=EDGE_LW,
                                   zorder=3))
    else:
        rate = (Q1_HI - Q1_LO - WINDOW_W) / N_TIME
        for t in scan:
            y = Q1_LO + WINDOW_W / 2 + rate * t
            ax.add_patch(Rectangle((t, y - WINDOW_W / 2),
                                   2 * half, WINDOW_W,
                                   facecolor=_face(ACTIVE, 0.55),
                                   edgecolor=EDGE, linewidth=EDGE_LW,
                                   zorder=3))

    ax.set_xticks([])
    ax.set_yticks([])
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_visible(False)


def _ms_row(ax):
    """Precursor ions as an MS1 spectrum, sharing the Q1 axis of the time row."""
    centre = (Q1_LO + Q1_HI) / 2
    ax.set_ylim(Q1_LO, Q1_HI)
    ax.set_xlim(0, 1.15)
    for m in BYSTANDERS:
        ax.plot([0, 0.55], [m, m], color='0.55', lw=1.3,
                solid_capstyle='butt')
    for dm, col, h in zip(PRECURSOR_DM, PRECURSORS, PRECURSOR_INT):
        ax.plot([0, h], [centre + dm, centre + dm], color=col, lw=2.0,
                solid_capstyle='butt')
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_visible(False)


def _bin_row(ax, mode):
    """The TOF bins one window's pulses fall into."""
    ax.set_xlim(Q1_LO, Q1_HI)
    ax.set_ylim(0, 1)
    w = (Q1_HI - Q1_LO) / N_BINS
    centre = (Q1_LO + Q1_HI) / 2
    for i in range(N_BINS):
        x = Q1_LO + i * w
        mid = x + w / 2
        if mode == 'swath':
            a = 0.75 if abs(mid - centre) <= WINDOW_W / 2 else 0.0
        else:
            a = 0.85 * float(np.exp(-0.5 * ((mid - centre) / 0.95) ** 2))
            a = a if a > 0.03 else 0.0
        if a <= 0:
            continue
        ax.add_patch(Rectangle((x + w * BIN_GAP / 2, 0.12),
                               w * (1 - BIN_GAP), 0.76,
                               facecolor=_face(ACTIVE, a),
                               edgecolor=EDGE, linewidth=EDGE_LW))

    ax.set_xticks([])
    ax.set_yticks([])
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_visible(False)


def _response_row(ax, mode):
    """One precursor's fragment signal against Q1 position."""
    ax.set_xlim(Q1_LO, Q1_HI)
    ax.set_ylim(0, 1.25)
    centre = (Q1_LO + Q1_HI) / 2
    q = np.linspace(Q1_LO, Q1_HI, 400)

    if mode == 'swath':
        y = np.where(np.abs(q - centre) <= WINDOW_W / 2, 1.0, 0.0)
        ax.fill_between(q, 0, y, color=ACTIVE, alpha=0.14, lw=0)
        lo = centre - WINDOW_W / 2
        third = WINDOW_W / len(PRECURSORS)
        v_third = 1.0 / len(PRECURSORS)
        for j, col in enumerate(PRECURSORS):
            x0 = lo + j * third
            ax.plot([x0, x0 + third], [1.0, 1.0], color=col, lw=2.4,
                    solid_capstyle='butt')
            y0 = j * v_third
            for x in (lo, lo + WINDOW_W):
                ax.plot([x, x], [y0, y0 + v_third], color=col, lw=2.4,
                        solid_capstyle='butt')
            ax.plot([centre + PRECURSOR_DM[j]], [1.12], marker='v',
                    ms=4, color=col, clip_on=False)
    else:
        for dx, col, h in zip(PRECURSOR_DM, PRECURSORS, PRECURSOR_INT):
            y = h * np.exp(-0.5 * ((q - centre - dx) / 1.05) ** 2)
            ax.plot(q, y, color=col, lw=1.8)
            ax.fill_between(q, 0, y, color=col, alpha=0.14, lw=0)
            ax.plot([centre + dx], [h + 0.12], marker='v', ms=4, color=col,
                    clip_on=False)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.8)


def panel(letter='a'):
    fig = plt.figure(figsize=(3.05, 2.60))
    gs = fig.add_gridspec(3, 4, height_ratios=[2.0, 0.42, 1.05],
                          width_ratios=[1.0, 0.30, 1.0, 0.30],
                          hspace=0.62, wspace=0.10,
                          left=0.145, right=0.995, top=0.875, bottom=0.11)
    swath, scan = _pulse_times()

    for col, mode, title in ((0, 'swath', 'Zeno SWATH DIA'),
                             (2, 'scan', 'ZT Scan DIA')):
        ax_t = fig.add_subplot(gs[0, col])
        _time_row(ax_t, mode, swath, scan)
        _ms_row(fig.add_subplot(gs[0, col + 1], sharey=ax_t))
        ax_t.text(0.62, 1.03, title, transform=ax_t.transAxes, ha='center',
                  va='bottom', fontsize=fs.LEGEND_FONTSIZE, fontweight='bold')
        ax_b = fig.add_subplot(gs[1, col:col + 2])
        _bin_row(ax_b, mode)
        ax_r = fig.add_subplot(gs[2, col:col + 2])
        _response_row(ax_r, mode)
        ax_t.set_xlabel('time', fontsize=fs.LEGEND_FONTSIZE, labelpad=1.5)
        ax_r.set_xlabel('Q1 dimension', fontsize=fs.LEGEND_FONTSIZE,
                        labelpad=1.5)
        if col == 0:
            ax_t.set_ylabel('Q1 position', fontsize=fs.LEGEND_FONTSIZE,
                            labelpad=2)
            ax_b.set_ylabel('TOF bins', fontsize=fs.LEGEND_FONTSIZE,
                            labelpad=2)
            ax_r.set_ylabel('signal', fontsize=fs.LEGEND_FONTSIZE,
                            labelpad=2)

    n_swath = sum(1 for _, k in swath
                  if Q1_LO + WINDOW_W * (k + 0.5) <= Q1_HI)
    print(f'    pulse period {PULSE_DT:g} in both halves over {N_TIME:g} '
          f'time units')
    print(f'    Zeno SWATH {len(swath)} pulses ({DWELL} per window, '
          f'{OVERHEAD:g} dead), ZT Scan {len(scan)} pulses '
          f'-> {len(scan) / len(swath):.2f}x')
    source = pd.DataFrame({
        'quantity': ['pulse_period', 'time_axis', 'pulses_per_window',
                     'overhead_per_window', 'pulses_zeno_swath',
                     'pulses_zt_scan'],
        'value': [PULSE_DT, N_TIME, DWELL, OVERHEAD, len(swath), len(scan)],
        'note': ['shared by both halves', 'shared by both halves',
                 'illustrative', 'illustrative', 'drawn', 'drawn']})
    fs.save_panel(fig, OUT, f'figure2{letter}_acquisition_schematic', source,
                  dpi=300)
    plt.close(fig)


def main(letter='a'):
    fs.init()
    print(f'panel {letter} -- acquisition schematic ...')
    panel(letter)


if __name__ == '__main__':
    main()
