"""Opening vendor raw files and caching them as HDF5."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

_READER_MODULE = {
    'thermo': 'alpharaw.thermo', 'thermo_raw': 'alpharaw.thermo',
    'sciex': 'alpharaw.sciex', 'sciex_wiff': 'alpharaw.sciex',
    'sciex_raw': 'alpharaw.sciex', 'mzml': 'alpharaw.mzml',
}
_EXT_TO_READER = {
    '.raw': 'thermo', '.wiff': 'sciex', '.mzml': 'mzml', '.mzml.gz': 'mzml',
}


def _mz_window(mz, mz_tol, mz_tol_unit):
    """(low, high) inclusive m/z bounds for one target."""
    if mz_tol_unit == 'ppm':
        half = mz * mz_tol * 1e-6
    elif mz_tol_unit in ('da', 'Da', 'th', 'Th'):
        half = mz_tol
    else:
        raise ValueError(f"mz_tol_unit must be 'ppm' or 'Da', got {mz_tol_unit!r}")
    return mz - half, mz + half


KEEP_ALL_PEAKS = 10 ** 12

DEFAULT_READER_OPTIONS = {'sciex': {'keep_k_peaks_per_spec': KEEP_ALL_PEAKS}}

PARSE_PEAKS_PER_GB = 4.4e8
PARSE_BYTES_PER_PEAK = 20
PARSE_MIN_FREE_GB = 4.0
PARSE_SAFETY = 1.1

def estimate_parse_memory(raw_path):
    """(GB needed, estimated peak count) to parse this raw file."""
    path = raw_path
    scan = raw_path + '.scan'
    if os.path.exists(scan):
        path = scan
    if os.path.isdir(path):
        size = sum(os.path.getsize(os.path.join(r, f))
                   for r, _, fs in os.walk(path) for f in fs)
    elif os.path.exists(path):
        size = os.path.getsize(path)
    else:
        return PARSE_MIN_FREE_GB, float('nan')
    peaks = size / 1e9 * PARSE_PEAKS_PER_GB
    need = peaks * PARSE_BYTES_PER_PEAK / 1e9 * PARSE_SAFETY
    return max(need, PARSE_MIN_FREE_GB), peaks

def _preflight_memory(raw_path, *, verbose=True, min_free_gb=None):
    """Refuse to start a parse that will not fit in the memory free right now."""
    try:
        import psutil
    except ImportError:
        return
    need, peaks = estimate_parse_memory(raw_path)
    if min_free_gb is not None:
        need = min_free_gb
    vm = psutil.virtual_memory()
    free, total = vm.available / 1e9, vm.total / 1e9
    if free >= need:
        if verbose:
            print(f'[raw] preflight: need ~{need:.0f} GB '
                  f'({peaks / 1e9:.1f}e9 peaks est.), {free:.0f} of '
                  f'{total:.0f} GB free')
        return
    fits = int(total // need) if need else 0
    raise MemoryError(
        f'{os.path.basename(raw_path)} needs ~{need:.0f} GB to parse '
        f'(~{peaks / 1e9:.1f} billion peaks estimated from its size) but only '
        f'{free:.0f} GB of {total:.0f} GB is free. This machine fits about '
        f'{fits} concurrent parse(s) of a file this size. Wait for a sibling '
        f'to finish, or pass min_free_gb= if the estimate is wrong. Nothing '
        f'was written.'
    )

def _cache_is_readable(cache_path):
    """True when the HDF opens and has content — a truncated cache does not."""
    try:
        import h5py
        with h5py.File(cache_path, 'r') as h:
            return len(h.keys()) > 0
    except ImportError:
        return True
    except Exception:
        return False


_PEAK_LIMIT_ATTR = 'peak_limit'

def _reader_opts(reader_type, reader_options):
    """Reader defaults with the caller's options layered on top."""
    opts = dict(DEFAULT_READER_OPTIONS.get(reader_type, {}))
    opts.update(reader_options or {})
    return opts

def _fmt_limit(value):
    """A peak cap as text, so 10**12 does not read as a real count."""
    return 'unlimited' if value == KEEP_ALL_PEAKS else f'{value:,}'

def _stamp_peak_limit(cache_path, limit):
    """Record the peak cap a cache was built under."""
    try:
        import h5py
        with h5py.File(cache_path, 'a') as h:
            h.attrs[_PEAK_LIMIT_ATTR] = -1 if limit is None else int(limit)
    except Exception:
        pass

NO_TRUNCATION = 10 ** 7


def _same_peak_limit(have, want):
    """Whether a cache built with cap ``have`` serves a request for ``want``."""
    if have is None or want is None:
        return have == want
    have, want = int(have), int(want)
    return have == want or (have >= NO_TRUNCATION and want >= NO_TRUNCATION)


def _cached_peak_limit(cache_path):
    """The cap a cache was built under, or None when it is not recorded."""
    try:
        import h5py
        with h5py.File(cache_path, 'r') as h:
            if _PEAK_LIMIT_ATTR not in h.attrs:
                return None
            return int(h.attrs[_PEAK_LIMIT_ATTR])
    except Exception:
        return None

def load_ms_data(raw_path, *, cache=True, cache_dir=None, centroided=True,
                 reader_type=None, reader_options=None, verbose=True,
                 min_free_gb=None):
    """Load a vendor raw file via alpharaw, caching it as HDF5.

    Parameters
      raw_path:     path to a .raw (Thermo) / .wiff (Sciex) / .mzML file, or an
                    existing .ms_data.hdf produced by a previous call.
      cache:        write/read the HDF cache (default True).
      cache_dir:    directory for the HDF cache; defaults to the raw file's dir.
      centroided:   centroid peaks on read (default True). Pass False to keep
                    the full profile (e.g. intact-protein native spectra, or
                    anything that measures peak SHAPE such as m/z resolution).
                    Note the Sciex reader ignores this and always returns
                    profile data — it warns that its centroiding is not well
                    implemented.
      reader_options: dict of attributes set on the alpharaw reader BEFORE the
                    raw file is read. This is how you reach reader settings the
                    alpharaw API does not expose as arguments. The one that
                    matters most: the Sciex reader defaults to
                    ``keep_k_peaks_per_spec = 2000``, i.e. it silently keeps
                    only the 2000 most abundant peaks of every spectrum. That
                    is invisible in the output and quietly truncates any
                    profile-shape or low-abundance analysis, so pass
                    ``reader_options={'keep_k_peaks_per_spec': 10**7}`` when
                    you need whole spectra.
      reader_type:  override the extension-based reader ('thermo' | 'sciex' |
                    'mzml'); normally inferred from the extension.
      min_free_gb:  override the pre-flight memory requirement. By default the
                    requirement is ESTIMATED from the file size (see
                    ``estimate_parse_memory``) and the parse is refused when
                    less than that is free — which also caps how many parses
                    can run at once, since free memory falls as siblings run.
                    Pass a number to override, or a very small one to disable
                    the check.
    """
    import os
    import importlib
    from alpharaw.ms_data_base import MSData_Base, ms_reader_provider

    ext = os.path.splitext(raw_path)[1].lower()

    if ext in ('.hdf', '.hdf5') or raw_path.lower().endswith('.ms_data.hdf'):
        data = MSData_Base(centroided=centroided)
        data.load_hdf(raw_path)
        if verbose:
            print(f'[raw] loaded HDF {raw_path}')
        return data

    if reader_type is None:
        reader_type = _EXT_TO_READER.get(ext)
        if reader_type is None:
            raise ValueError(
                f'cannot infer reader from extension {ext!r}; pass '
                f'reader_type= one of {sorted(set(_READER_MODULE))}'
            )

    stem = os.path.basename(raw_path)
    stem = stem[:-len(ext)] if ext else stem
    cdir = cache_dir if cache_dir is not None else os.path.dirname(os.path.abspath(raw_path))
    cache_path = os.path.join(cdir, stem + '.ms_data.hdf')

    if cache and os.path.exists(cache_path) and \
            os.path.getmtime(cache_path) >= os.path.getmtime(raw_path):
        want = _reader_opts(reader_type, reader_options).get(
            'keep_k_peaks_per_spec')
        have = _cached_peak_limit(cache_path)
        stale = want is not None and not _same_peak_limit(have, want)
        if stale:
            had = ('unrecorded (alpharaw default 2,000)' if have is None
                   else _fmt_limit(have))
            print(f'[raw] cache holds a per-spectrum peak limit of {had}, not '
                  f'{_fmt_limit(want)} — re-parsing rather than returning '
                  f'truncated peaks: {cache_path}')
        elif _cache_is_readable(cache_path):
            data = MSData_Base(centroided=centroided)
            data.load_hdf(cache_path)
            if verbose:
                print(f'[raw] loaded cache {cache_path}')
            return data
        else:
            print(f'[raw] cache is unreadable, re-parsing: {cache_path}')
        try:
            os.remove(cache_path)
        except OSError as exc:
            raise RuntimeError(
                f'cache {cache_path} could not be removed ({exc}); delete it '
                f'and retry'
            ) from exc

    _preflight_memory(raw_path, verbose=verbose, min_free_gb=min_free_gb)

    mod = importlib.import_module(_READER_MODULE[reader_type])
    if hasattr(mod, 'register_readers'):
        mod.register_readers()
    reader = ms_reader_provider.get_reader(reader_type, centroided=centroided)
    if reader is None:
        raise RuntimeError(
            f'alpharaw has no {reader_type!r} reader available (vendor '
            f'dependencies missing?)'
        )
    opts = _reader_opts(reader_type, reader_options)
    for key, value in opts.items():
        if not hasattr(reader, key):
            if key in DEFAULT_READER_OPTIONS.get(reader_type, {}):
                continue
            raise AttributeError(
                f'{type(reader).__name__} has no attribute {key!r}; '
                f'reader_options must name real reader settings'
            )
        setattr(reader, key, value)
        if verbose:
            print(f'[raw] reader.{key} = {_fmt_limit(value)}')
    if verbose:
        print(f'[raw] reading {raw_path} ({reader_type})')
    reader.import_raw(raw_path)

    if cache:
        os.makedirs(cdir, exist_ok=True)
        reader.save_hdf(cache_path)
        _stamp_peak_limit(cache_path, opts.get('keep_k_peaks_per_spec'))
        if verbose:
            print(f'[raw] cached -> {cache_path}')
    return reader
