"""Acquisition metadata from a vendor raw file, without decoding any peaks."""

from __future__ import annotations

import glob as _glob
import os

import numpy as np
import pandas as pd

__all__ = [
    'read_acquisition_metadata',
    'survey_raw_files',
    'acquisition_times',
]

_SCANNING_MIN_WINDOWS = 200


def _open_wiff(path):
    """Open a wiff batch and return (reader, sample, mass_spectrometer_sample)."""
    from alpharaw.raw_access.pysciexwifffilereader import WillFileReader

    reader = WillFileReader(path)
    sample = reader._wiff_file.GetSample(0)
    return reader, sample, sample.MassSpectrometerSample


def _experiment_rows(ms_sample):
    """One dict per MS experiment (the method's scan definitions)."""
    rows = []
    for i in range(ms_sample.ExperimentCount):
        details = ms_sample.GetMSExperiment(i).Details
        mass_ranges = details.MassRangeInfo
        first = mass_ranges[0] if len(mass_ranges) else None
        rows.append({
            'experiment': i,
            'experiment_type': str(details.ExperimentType),
            'name': str(details.Name),
            'n_scans': int(details.NumberOfScans),
            'start_mz': float(details.StartMass),
            'end_mz': float(details.EndMass),
            'polarity': str(details.Polarity),
            'resolution': float(details.DefaultResolution),
            'is_swath': bool(details.IsSwath),
            'isolation_width': float(getattr(first, 'IsolationWindow', np.nan))
            if first is not None else np.nan,
            'dwell_time_ms': float(getattr(first, 'DwellTime', np.nan))
            if first is not None else np.nan,
        })
    return rows


def _span(widths, centres):
    """Precursor range the isolation windows cover end to end, in Th."""
    return float(centres.max() - centres.min()) + float(np.nanmedian(widths))


def _coverage(widths, centres):
    """Summed window width over the spanned precursor range. ~1 when tiled."""
    if widths.size == 0 or centres.size < 2 or np.isnan(widths).all():
        return np.nan
    span = _span(widths, centres)
    return float(np.nansum(widths)) / span if span > 0 else np.nan


def _classify(swath_flags, n_windows):
    """Scan mode, from the method's declared parameters."""
    if n_windows == 0:
        return 'ms1'
    if not any(swath_flags):
        return 'targeted'
    return 'scanning_dia' if n_windows >= _SCANNING_MIN_WINDOWS else 'dia'


def read_acquisition_metadata(path, *, with_experiments=False):
    """Method and instrument metadata for one raw file, without reading peaks."""
    ext = os.path.splitext(path)[1].lower()
    if ext != '.wiff':
        raise NotImplementedError(
            f'read_acquisition_metadata supports Sciex .wiff only, got {ext!r}. '
            'Thermo .raw and mzML expose their method metadata through their '
            'own readers; add a branch here when one is needed.')

    reader, sample, ms_sample = _open_wiff(path)
    try:
        details = sample.Details
        exps = _experiment_rows(ms_sample)

        first = ms_sample.GetMSExperiment(0)
        n_cycles = int(first.Details.NumberOfScans)
        rt_start = float(first.GetRTFromExperimentCycle(0))
        rt_end = float(first.GetRTFromExperimentCycle(max(n_cycles - 1, 0)))

        products = [e for e in exps if e['experiment_type'] != 'MS']
        widths = np.array([e['isolation_width'] for e in products], dtype=float)
        centres = np.array([c for c in (_centre_from_name(e['name'])
                                        for e in products) if c is not None])

        meta = {
            'file': os.path.basename(path),
            'path': os.path.abspath(path),
            'sample_name': str(details.SampleName),
            'vial': str(details.Vial),
            'method': os.path.basename(str(details.AcquisitionMethodName)),
            'acquired': pd.Timestamp(details.AcquisitionDateTime.ToString('O')),
            'instrument': str(details.InstrumentName),
            'serial': str(details.InstrumentSerialNumber),
            'software': str(details.SoftwareVersion),
            'batch': str(details.BatchName),
            'operator': str(details.UserName),
            'n_experiments': len(exps),
            'n_windows': len(products),
            'n_cycles': n_cycles,
            'rt_start_min': rt_start,
            'rt_end_min': rt_end,
            'cycle_time_s': ((rt_end - rt_start) / (n_cycles - 1) * 60.0
                             if n_cycles > 1 else np.nan),
            'ms1_start_mz': exps[0]['start_mz'] if exps else np.nan,
            'ms1_end_mz': exps[0]['end_mz'] if exps else np.nan,
            'polarity': exps[0]['polarity'] if exps else '',
            'dwell_time_ms': float(np.nanmedian(
                [e['dwell_time_ms'] for e in exps])) if exps else np.nan,
            'isolation_width_median': (float(np.nanmedian(widths))
                                       if widths.size else np.nan),
            'isolation_width_min': (float(np.nanmin(widths))
                                    if widths.size else np.nan),
            'isolation_width_max': (float(np.nanmax(widths))
                                    if widths.size else np.nan),
            'precursor_min_mz': float(centres.min()) if centres.size else np.nan,
            'precursor_max_mz': float(centres.max()) if centres.size else np.nan,
            'is_swath': bool(any(e['is_swath'] for e in products)),
            'window_coverage': _coverage(widths, centres),
            'scan_mode': _classify([e['is_swath'] for e in products],
                                   len(products)),
        }
        if with_experiments:
            meta['experiments'] = pd.DataFrame(exps)
        return meta
    finally:
        reader.close()


def _centre_from_name(name):
    """Precursor centre m/z from a product-ion experiment's display name."""
    marker = ' of '
    if marker not in name:
        return None
    tail = name.split(marker, 1)[1].split('(')[0].strip()
    parts = tail.split(' to ')
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    return float(np.mean(values))


def survey_raw_files(paths, *, verbose=True, errors='warn'):
    """Metadata table for many raw files — one row per run."""
    paths = _resolve(paths)
    rows = []
    for i, path in enumerate(paths, 1):
        try:
            rows.append(read_acquisition_metadata(path))
        except Exception as exc:
            if errors == 'raise':
                raise
            rows.append({'file': os.path.basename(path),
                         'path': os.path.abspath(path),
                         'error': f'{type(exc).__name__}: {exc}'})
        if verbose:
            print(f'  survey_raw_files: {i}/{len(paths)} {rows[-1]["file"]}',
                  flush=True)
    df = pd.DataFrame(rows)
    if 'acquired' in df.columns:
        df = df.sort_values('acquired', kind='stable').reset_index(drop=True)
    return df


def read_acquisition_time(path):
    """The instrument's ``AcquisitionDateTime`` for one wiff, and nothing else."""
    ext = os.path.splitext(path)[1].lower()
    if ext != '.wiff':
        raise NotImplementedError(
            f'read_acquisition_time supports Sciex .wiff only, got {ext!r}.')

    reader, sample, _ = _open_wiff(path)
    try:
        return pd.Timestamp(sample.Details.AcquisitionDateTime.ToString('O'))
    finally:
        close = getattr(reader, 'close', None)
        if close is not None:
            try:
                close()
            except Exception:
                pass


def acquisition_times(paths, *, verbose=False):
    """Instrument-written acquisition timestamp per run, keyed by file stem."""
    paths = _resolve(paths)
    out = {}
    for path in paths:
        try:
            when = read_acquisition_time(path)
        except Exception:
            continue
        out[os.path.splitext(os.path.basename(path))[0]] = when
        if verbose:
            print(f'  {os.path.basename(path)}: {out[list(out)[-1]]}')
    return pd.Series(out, name='acquired').sort_values()


def _resolve(paths):
    """Accept a directory, a glob, or a list of paths; return a sorted list."""
    if isinstance(paths, (str, os.PathLike)):
        paths = str(paths)
        if os.path.isdir(paths):
            paths = os.path.join(paths, '*.wiff')
        paths = sorted(_glob.glob(paths))
    return [str(p) for p in paths]
