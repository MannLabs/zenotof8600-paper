"""In-silico digestion, sequence properties and FASTA loading."""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd


_PROTEASE_SITES = {
    'trypsin': 'RK',
    'lysc': 'K',
    'argc': 'R',
    'chymotrypsin': 'FWY',
    'gluc': 'ED',
}


def count_missed_cleavages(sequence, protease='trypsin'):
    """Count missed cleavages within `sequence` (excluding the C-terminus)."""
    if not sequence:
        return 0
    sites = _PROTEASE_SITES.get(protease.lower(), 'RK')
    return sum(1 for aa in sequence[:-1] if aa in sites)


_KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}


