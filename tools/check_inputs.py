"""Check that every input of every figure is where the code expects it."""

import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import zt8600_config as cfg

EXPECTED = os.path.join(REPO, 'tools', 'expected_inputs.csv')
ORDER = [f'figure{i}' for i in range(1, 7)] + \
    [f'supplementary_figure{i}' for i in range(1, 9)]


def check(figures=None, raw=False):
    """One row per expected file with ``present`` and ``expected_in``."""
    t = pd.read_csv(EXPECTED)
    if figures:
        t = t[t['figure'].isin(figures)]
    if not raw:
        t = t[t['use'] == 'panels']
    root = {'data': cfg.DATA_ROOT, 'raw': cfg.RAW_ROOT}
    full = [os.path.join(root[r], *p.split('/')) for r, p in zip(t['root'], t['path'])]
    t = t.assign(full_path=full,
                 present=[os.path.exists(f) for f in full],
                 expected_in=[os.path.dirname(f) for f in full])
    return t


def summary(t):
    """Per figure: files expected, missing, and whether it can be built."""
    g = t.groupby(['figure', 'use'])
    s = pd.DataFrame({'expected': g.size(), 'missing': g['present'].apply(lambda p: (~p).sum())})
    s = s.reset_index().pivot(index='figure', columns='use', values='missing').fillna(0).astype(int)
    s['status'] = ['complete' if r.sum() == 0 else 'missing inputs' for _, r in s.iterrows()]
    return s.reindex([f for f in ORDER if f in s.index])


def report(t, max_list=8):
    print(f'data package (ZT8600_DATA_ROOT): {cfg.DATA_ROOT}  '
          f'{"found" if os.path.isdir(cfg.DATA_ROOT) else "NOT FOUND"}')
    if (t['root'] == 'raw').any():
        print(f'raw deposit  (ZT8600_RAW_ROOT):  {cfg.RAW_ROOT}  '
              f'{"found" if os.path.isdir(cfg.RAW_ROOT) else "NOT FOUND"}')
    print()
    print(summary(t).to_string())
    for fig, g in t[~t['present']].groupby('figure'):
        print(f'\n{fig}: {len(g)} file(s) missing')
        for folder, h in list(g.groupby('expected_in'))[:max_list]:
            names = h['path'].map(lambda p: p.split('/')[-1]).tolist()
            more = f' (+{len(names) - 3} more)' if len(names) > 3 else ''
            print(f'  expected in {folder}:\n    ' + ', '.join(names[:3]) + more)
    ok = t['present'].all()
    print('\nall inputs present' if ok else
          '\nfigures with missing inputs are skipped by tools/run_figures.py; '
          'the others build normally')
    return ok


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    sys.exit(0 if report(check(args or None, raw='--raw' in sys.argv)) else 1)
