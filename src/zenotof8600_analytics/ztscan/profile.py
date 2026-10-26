"""Locating a fragment's maximum on its Q1 profile."""

import numpy as np


def peak_index(rows):
    """Sub-sample position of each row's maximum, by parabolic interpolation."""
    out = np.zeros(len(rows))
    for i, y in enumerate(rows):
        k = int(np.argmax(y))
        if 0 < k < len(y) - 1:
            a, b, c = y[k - 1], y[k], y[k + 1]
            d = a - 2 * b + c
            out[i] = k + (0.5 * (a - c) / d if d != 0 else 0.0)
        else:
            out[i] = float(k)
    return out
