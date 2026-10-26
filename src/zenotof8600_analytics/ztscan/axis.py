"""The Q1 axis: a quadrupole setpoint, not a mass measurement."""

from __future__ import annotations

import json
import os

import numpy as np


def q1_correction(axis, mz, rt):
    """The Q1 mass-axis correction, for either form of calibration."""
    mz = np.asarray(mz, float)
    rt = np.asarray(rt, float)
    if axis is None:
        return np.zeros(np.broadcast(mz, rt).shape)
    if isinstance(axis, dict):
        return (np.interp(mz, axis['mz_knots'], axis['mz_corr'])
                + axis['slope_rt'] * rt)
    a, b = axis[0], axis[1]
    c = axis[2] if len(axis) > 2 else 0.0
    return a + b * mz + c * rt


