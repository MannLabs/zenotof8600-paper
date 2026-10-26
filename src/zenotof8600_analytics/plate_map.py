"""Plate-map metadata for MS experiments."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from itertools import product
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb


_DEFAULT_PALETTE = ['#FA7A61', '#B178E2', '#78B8E2', '#6BC88F', '#F5C842', '#E85D9A']

_PLATE_DIMENSIONS: dict[int, tuple[str, tuple[int, ...]]] = {
    96:  ('ABCDEFGH',         tuple(range(1, 13))),
    384: ('ABCDEFGHIJKLMNOP', tuple(range(1, 25))),
}

_WELL_RE = re.compile(r'([A-P])(\d+)')


_WELL_TOKEN_RE = re.compile(r'^([A-Pa-p])([0-9]{1,2})$')
_QUADRANT_TOKEN_RE = re.compile(r'^[Qq]([0-9]{1,2})$')

PLATE_FIELDS = ('plate_group', 'plate_well', 'plate_row', 'plate_col',
                'plate_quadrant')


def _is_well(token):
    m = _WELL_TOKEN_RE.match(token)
    if m is None:
        return None
    col = int(m.group(2))
    return (m.group(1).upper(), col) if 1 <= col <= 24 else None


def _derived_quadrant(row, col):
    """384 -> 96 quadrant of a well, by the standard interleaved mapping."""
    r = 'ABCDEFGHIJKLMNOP'.index(row) + 1
    if r <= 8 and col <= 12:
        return None
    return f'Q{2 * (r % 2 == 0) + (col % 2 == 0) + 1}'


def parse_run_position(name):
    """Plate position parsed from the tail of a run / sample name."""
    empty = dict.fromkeys(PLATE_FIELDS)
    tokens = [t for t in re.split(r'[_\-]', str(name)) if t]
    for i in range(len(tokens) - 1, -1, -1):
        well = _is_well(tokens[i])
        if well is None:
            continue
        row, col = well
        group = None
        if i > 0:
            prev = tokens[i - 1]
            if (_QUADRANT_TOKEN_RE.match(prev) or _is_well(prev)
                    or re.match(r'^[Pp][0-9]{1,2}$', prev)):
                group = prev
        quadrant = (f'Q{_QUADRANT_TOKEN_RE.match(group).group(1)}'
                    if group and _QUADRANT_TOKEN_RE.match(group)
                    else _derived_quadrant(row, col))
        return {'plate_group': group, 'plate_well': f'{row}{col}',
                'plate_row': row, 'plate_col': col,
                'plate_quadrant': quadrant}
    return empty


def plate_metadata(runs):
    """DataFrame of `PLATE_FIELDS` for a sequence of run names, indexed by run."""
    runs = list(runs)
    return pd.DataFrame([parse_run_position(r) for r in runs],
                        index=pd.Index(runs, name='run'),
                        columns=list(PLATE_FIELDS))
