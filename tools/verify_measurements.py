"""Re-measure every raw-derived table of the data package and compare it with the published one."""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import zt8600_config as cfg

RTOL = 1e-6

STEPS = {
    'ion_current': (['figure1/ion_current/*.csv'],
                    ('figure1', 'measure_ion_current.py', [])),
    'rt_q1_map': (['H032_E376/ztscan_rt_q1_map_A9_grid.npz'],
                  ('figure2', 'measure_rt_q1_map.py', ['A9'])),
    'q1_axes': (['H032_E376/q1_calibration_old/q1_axis_*.json'],
                ('figure2', 'fit_provisional_q1_axes.py', [])),
    'q1_wholefile': (['H032_E376/q1_wholefile/*.csv'],
                     ('figure2', 'measure_q1_precision_wholefile.py', ['--runs', 'A5,A9,A10,B1,B5,B9'])),
    'q1_axis': (['H032_E376/q1_axis_*wholefile.json'],
                ('figure2', 'fit_q1_axis_wholefile.py', [])),
    'q1_window': (['H032_E376/q1_window_A9_wholefile.npz'],
                  [('figure2', 'fit_q1_axis_wholefile.py', []),
                   ('figure2', 'extract_q1_window.py', ['--axis', 'wholefile'])]),
    'q1_axis_transfer': (['H032_E376/q1_axis_transfer_A9_to_A10.csv'],
                         ('figure2', 'q1_axis_transfer.py', [])),
    'mass_accuracy_8600': (['H032_E352/*.rawdiag.parquet'],
                           ('supplementary_figure1', 'measure_mass_accuracy_resolution.py', ['8600'])),
    'mass_accuracy_7600': (['H032_E303/*.rawdiag.parquet'],
                           ('supplementary_figure1', 'measure_mass_accuracy_resolution.py', ['7600'])),
    'lc_peak_width': (['H032_E350/lc_peak_width_sourcedata.csv'],
                      ('supplementary_figure5', 'measure_lc_peak_width.py', [])),
    '6x5': (['H032_E396/6x5_*'], ('figure6', 'measure_6x5.py', [])),
    '6x5_ms1': (['H032_E396/ms1/6x5_*'], ('figure6', 'measure_6x5_ms1.py', [])),
    'transition_check': (['H032_E397/transition_check/*'],
                         ('figure6', 'occupancy_transition_check.py', [])),
    'light_site_ions': (['H032_E397/light_site_ions.csv'],
                        ('figure6', 'light_site_ions.py', [])),
    'ps129_ztscan': (['H032_E397/ztscan_ps129_occupancy.csv'],
                     ('supplementary_figure7', 'measure_ps129_ztscan.py', [])),
    'ztscan_site_ions': (['H032_E397/ztscan_site_ions.csv'],
                         ('supplementary_figure7', 'ztscan_site_ions.py', [])),
    'summed_spectra': (['H032_E391/summed_spectra/*'], ('figure6', 'figure6.py', [])),
    'standard_6x5': (['H032_E396/standard_*.csv', 'H032_E396/trace_table_*.csv'],
                     ('supplementary_figure6', 'supplementary_figure6.py', [])),
    'q1_resolution': (['H032_E276/q1_resolution_*.csv'],
                      ('supplementary_figure6', 'q1_resolution.py', [])),
    'acquired_windows': (['supplementary_figure8/acquired_windows_E352_A1.csv'],
                         ('supplementary_figure8', 'extract_acquired_windows.py', [])),
}


def tables(root, patterns):
    return sorted({os.path.relpath(f, root).replace(os.sep, '/')
                   for p in patterns for f in glob.glob(os.path.join(root, p))
                   if os.path.isfile(f)})


def _num_compare(a, b):
    """'' if equal, ('rel', max relative deviation) if within RTOL, else a reason."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return f'shape {a.shape} vs {b.shape}'
    if np.array_equal(a, b, equal_nan=True):
        return ''
    if not np.allclose(a, b, rtol=RTOL, atol=1e-12, equal_nan=True):
        d = np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))
        return f'values differ (max relative deviation {d:.3g})'
    return ('rel', float(np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))))


def compare(new, old):
    """'' identical, 'within tolerance ...', or a description of the difference."""
    ext = os.path.splitext(old)[1].lower()
    try:
        if ext in ('.csv', '.parquet'):
            r = pd.read_csv if ext == '.csv' else pd.read_parquet
            a, b = r(new), r(old)
            if list(a.columns) != list(b.columns):
                return f'columns differ: {sorted(set(a.columns) ^ set(b.columns))}'
            if len(a) != len(b):
                return f'{len(a)} rows vs {len(b)}'
            worst = 0.0
            for c in b.columns:
                if pd.api.types.is_numeric_dtype(b[c]) and pd.api.types.is_numeric_dtype(a[c]):
                    res = _num_compare(a[c], b[c])
                    if isinstance(res, tuple):
                        worst = max(worst, res[1])
                    elif res:
                        return f'{c}: {res}'
                elif not a[c].astype(str).equals(b[c].astype(str)):
                    return f'{c}: text differs'
            return f'within tolerance (max {worst:.2g})' if worst else ''
        if ext == '.npz':
            a, b = np.load(new, allow_pickle=True), np.load(old, allow_pickle=True)
            if sorted(a.files) != sorted(b.files):
                return 'arrays differ'
            for k in b.files:
                res = _num_compare(a[k], b[k]) if b[k].dtype.kind in 'fiub' else (
                    '' if np.array_equal(a[k], b[k]) else f'{k} differs')
                if res and not isinstance(res, tuple):
                    return f'{k}: {res}'
            return ''
        if ext == '.json':
            a, b = json.load(open(new)), json.load(open(old))
            if a == b:
                return ''
            fa, fb = _floats(a), _floats(b)
            if len(fa) != len(fb):
                return 'structure differs'
            res = _num_compare(fa, fb)
            return f'within tolerance (max {res[1]:.2g})' if isinstance(res, tuple) else (
                res or 'non-numeric fields differ')
        same = open(new, 'rb').read() == open(old, 'rb').read()
        return '' if same else 'bytes differ'
    except Exception as e:
        return f'could not compare: {e}'


def _floats(x):
    if isinstance(x, dict):
        return [v for k in sorted(x) for v in _floats(x[k])]
    if isinstance(x, list):
        return [v for i in x for v in _floats(i)]
    return [float(x)] if isinstance(x, (int, float)) and not isinstance(x, bool) else []


def run_step(name, scratch_data, scratch_out):
    env = {**os.environ, 'ZT8600_DATA_ROOT': scratch_data,
           'ZT8600_OUTPUT_ROOT': scratch_out, 'PYTHONIOENCODING': 'utf-8',
           'MPLBACKEND': 'Agg'}
    spec = STEPS[name][1]
    specs = spec if isinstance(spec, list) else [spec]
    cmds = [([sys.executable, script, *args], os.path.join(REPO, folder))
            for folder, script, args in specs]
    t0 = time.time()
    for cmd, cwd in cmds:
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                           encoding='utf-8', errors='replace')
        if r.returncode:
            print(r.stdout[-1200:], r.stderr[-2500:], sep='\n')
            break
    return r.returncode == 0, time.time() - t0


def main(names, scratch):
    names = names or list(STEPS)
    data = cfg.DATA_ROOT
    scratch_data = os.path.join(scratch, 'data')
    scratch_out = os.path.join(scratch, 'output')
    removed = {n: tables(data, STEPS[n][0]) for n in names}
    if not os.path.isdir(scratch_data):
        print(f'copying the data package to {scratch_data} ...', flush=True)
        shutil.copytree(data, scratch_data)
    for n in names:
        for t in removed[n]:
            p = os.path.join(scratch_data, t)
            if os.path.exists(p):
                os.remove(p)
    report = []
    for n in names:
        if not removed[n]:
            print(f'{n}: no such table in the data package, skipped')
            continue
        ok, secs = run_step(n, scratch_data, scratch_out)
        for t in removed[n]:
            new = os.path.join(scratch_data, t)
            res = ('not regenerated' if not os.path.exists(new)
                   else compare(new, os.path.join(data, t)))
            report.append({'step': n, 'table': t, 'ran': ok,
                           'result': res or 'identical'})
        print(f'{"ok  " if ok else "FAIL"} {n:20s} {secs:7.0f} s  '
              + ', '.join(f'{r["result"]}' for r in report if r['step'] == n)[:200],
              flush=True)
    rep = pd.DataFrame(report)
    out = os.path.join(scratch, 'verification_report.csv')
    rep.to_csv(out, index=False)
    bad = rep[~rep['result'].str.startswith(('identical', 'within'))]
    print(f'\n{len(rep)} tables: {len(rep) - len(bad)} reproduce, {len(bad)} do not')
    if len(bad):
        print(bad.to_string(index=False))
    print('report:', out)
    print(f're-measured data package: {scratch_data} (build the figures from it '
          f'with ZT8600_DATA_ROOT pointing there)')
    return bad.empty


if __name__ == '__main__':
    args = sys.argv[1:]
    scratch = os.path.join(cfg.OUTPUT_ROOT, 'verify')
    if '--scratch' in args:
        i = args.index('--scratch')
        scratch = args[i + 1]
        del args[i:i + 2]
    sys.exit(0 if main([a for a in args if not a.startswith('--')], scratch) else 1)
