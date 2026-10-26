"""Build every panel and composite figure."""

import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENTRY_POINTS = {
    'figure1': ['figure1.py'],
    'figure2': ['figure2.py'],
    'figure3': ['figure3.py'],
    'figure4': ['single_cell.py', 'cell_cycle.py',
                'rank_plot.py', 'dvp_tissue.py'],
    'figure5': ['figure5.py'],
    'figure6': ['figure6.py'],
    'supplementary_figure1': ['supplementary_figure1.py'],
    'supplementary_figure2': ['supplementary_figure2.py', 'q1_abundance_occupancy.py'],
    'supplementary_figure3': ['supplementary_figure3.py'],
    'supplementary_figure4': ['supplementary_figure4.py'],
    'supplementary_figure5': ['supplementary_figure5.py', 'fraction_sets.py'],
    'supplementary_figure6': ['q1_resolution.py', 'supplementary_figure6.py'],
    'supplementary_figure7': ['supplementary_figure7.py'],
    'supplementary_figure8': ['supplementary_figure8.py'],
}


def run(figure, script, *args):
    t0 = time.time()
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'MPLBACKEND': 'Agg'}
    r = subprocess.run([sys.executable, script, *args], cwd=os.path.join(REPO, figure),
                       env=env, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    ok = r.returncode == 0
    print(f'{"ok  " if ok else "FAIL"} {figure}/{script}  {time.time() - t0:6.1f} s',
          flush=True)
    if not ok:
        print(r.stdout[-1500:], r.stderr[-3000:], sep='\n')
    return ok


def main(figures):
    """Every script runs whatever the others do; composites are built from whatever panels exist, so missing inputs cost only their own figures."""
    figures = figures or list(ENTRY_POINTS)
    results = {(f, s): run(f, s) for f in figures for s in ENTRY_POINTS[f]}
    mains = [f[len('figure'):] for f in figures if f.startswith('figure')]
    if any(f.startswith('supplementary') for f in figures):
        run('.', 'make_supplementary_composites.py')
    if mains:
        run('.', 'make_main_figures.py', *mains)
    failed = [f'{f}/{s}' for (f, s), ok in results.items() if not ok]
    print(f'\n{len(results) - len(failed)} of {len(results)} scripts ran'
          + (f'; failed: {", ".join(failed)} (tools/check_inputs.py lists '
             f'what is missing)' if failed else ''))
    return not failed


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1:]) else 1)
