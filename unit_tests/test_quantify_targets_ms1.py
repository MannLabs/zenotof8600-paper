"""quantify_targets(ms_level=1): the MS1 survey through the MS2 machinery."""
import numpy as np
import pandas as pd
import pytest

from zenotof8600_analytics.raw import targeted

Z = 2
MONO = 500.0
ISO = [MONO + k * targeted.ISOTOPE_SPACING_DA / Z for k in range(3)]
ISO_REL = [1.0, 0.6, 0.25]
FRAG = 650.0


class _MS:

    def __init__(self, n=120, apex=60, sigma=3.0, noise=1.0):
        rng = np.random.default_rng(1)
        shift = 2.0 * 2 / Z
        rows, peaks, start = [], [], 0
        for i in range(n):
            g = np.exp(-0.5 * ((i - apex) / sigma) ** 2)
            for level in (1, 2):
                if level == 1:
                    mz, it = [], []
                    for amt, off in ((1e5, 0.0), (1e4, shift)):
                        for m, rel in zip(ISO, ISO_REL):
                            grid = (m + off) * (1 + np.linspace(-90, 90, 25)
                                                / 1e6)
                            mz.append(grid)
                            it.append(amt * rel * g * np.exp(
                                -4 * np.log(2) * ((grid - m - off)
                                                  / ((m + off) * 30e-6))
                                ** 2))
                    mz, it = np.concatenate(mz), np.concatenate(it)
                    prec = np.nan
                else:
                    grid = FRAG * (1 + np.linspace(-90, 90, 25) / 1e6)
                    mz = grid
                    it = 5e4 * g * np.exp(-4 * np.log(2) * ((grid - FRAG)
                                                           / (FRAG * 30e-6))
                                          ** 2)
                    prec = MONO
                it = it + rng.uniform(0, noise, it.size)
                o = np.argsort(mz)
                peaks.append(pd.DataFrame({'mz': mz[o], 'intensity': it[o]}))
                rows.append({'rt': i * 0.01 + (0 if level == 1 else 0.005),
                             'ms_level': level, 'precursor_mz': prec,
                             'isolation_lower_mz': prec - 1 if level == 2
                             else np.nan,
                             'isolation_upper_mz': prec + 1 if level == 2
                             else np.nan,
                             'peak_start_idx': start,
                             'peak_stop_idx': start + mz.size})
                start += mz.size
        self.spectrum_df = pd.DataFrame(rows)
        self.peak_df = pd.concat(peaks, ignore_index=True)


def _ms1_transitions():
    rows = []
    for t, off in ((0, 0.0), (1, 2.0 * 2 / Z)):
        for k, m in enumerate(ISO):
            rows.append({'target_idx': t, 'sequence': 'PEPTIDEK', 'charge': Z,
                         'precursor_mz': MONO + off, 'ion_type': 'M',
                         'ion_number': k, 'fragment_charge': Z,
                         'product_mz': m + off})
    return pd.DataFrame(rows)


def test_ms1_quantifies_the_isotope_envelope():
    out = targeted.quantify_targets(
        {'r1': _MS()}, _ms1_transitions(), ms_level=1, mz_tol=30.0,
        decoy_offset_da=6.7, verbose=False)
    assert (out['ms_level'] == 1).all()
    a = out.set_index('target_idx')['area']
    assert a[0] / a[1] == pytest.approx(10.0, rel=0.05)
    assert out['reference_target_idx'].nunique() == 1
    assert (out['trace_r'] > 0.95).all()
    assert (out['signal_to_decoy'] > 100).all()


def test_ms1_ignores_the_default_fragment_filter():
    out = targeted.quantify_targets(
        {'r1': _MS()}, _ms1_transitions(), ms_level=1, mz_tol=30.0,
        decoy_offset_da=None, verbose=False)
    assert (out['area'] > 0).all()


def test_ms2_default_unchanged_and_labelled():
    frags = pd.DataFrame({'target_idx': [0], 'sequence': ['PEPTIDEK'],
                          'charge': [Z], 'precursor_mz': [MONO],
                          'product_mz': [FRAG]})
    out = targeted.quantify_targets(
        {'r1': _MS()}, frags, fragment_filter=None, match='window',
        mz_tol=30.0, decoy_offset_da=None, verbose=False)
    assert (out['ms_level'] == 2).all()
    assert out['area'].iloc[0] > 0


def test_bad_ms_level_raises():
    with pytest.raises(ValueError, match='ms_level'):
        targeted.quantify_targets({'r1': _MS()}, _ms1_transitions(),
                                  ms_level=3, verbose=False)


def test_precursor_isotope_transitions():
    t = pd.DataFrame({'sequence': ['LGFTDLFSK'], 'mods': [''],
                      'mod_sites': [''], 'charge': [2], 'level': [5]},
                     index=[7])
    tr = targeted.precursor_isotope_transitions(t, n_isotopes=3)
    assert list(tr['target_idx']) == [7, 7, 7]
    assert list(tr['ion_number']) == [0, 1, 2]
    d = np.diff(tr['product_mz'])
    assert np.allclose(d, targeted.ISOTOPE_SPACING_DA / 2)
    assert (tr['level'] == 5).all()
    assert tr['product_mz'].iloc[0] == pytest.approx(514.2766, abs=2e-3)
