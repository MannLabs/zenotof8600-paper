"""Shim: the merging logic now lives in the library, not here."""

from zenotof8600_analytics.raw import merge_peaks, profile_tolerance_ok

__all__ = ['merge_profile', 'merge_peaks', 'profile_tolerance_ok',
           'DEFAULT_TOL_PPM']

DEFAULT_TOL_PPM = 1.0


def merge_profile(mz, intensity, tol_ppm=DEFAULT_TOL_PPM):
    """Profile-sample merge: median position, summed intensity."""
    return merge_peaks(mz, intensity, tol=tol_ppm, tol_unit='ppm',
                       position='median')
