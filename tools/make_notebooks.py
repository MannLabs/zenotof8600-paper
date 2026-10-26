"""Write the notebooks in notebooks/ from the scripts' entry points."""

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'notebooks')

ENTRY = {
    'figure1': [('figure1.py', False)],
    'figure2': [('figure2.py', False)],
    'figure3': [('figure3.py', True)],
    'figure4': [('single_cell.py', False), ('cell_cycle.py', False),
                ('rank_plot.py', False),
                ('dvp_tissue.py', False)],
    'figure5': [('figure5.py', True)],
    'figure6': [('figure6.py', False)],
    'supplementary_figure1': [('supplementary_figure1.py', True)],
    'supplementary_figure2': [('supplementary_figure2.py', False),
                              ('q1_abundance_occupancy.py', True)],
    'supplementary_figure3': [('supplementary_figure3.py', True)],
    'supplementary_figure4': [('supplementary_figure4.py', False)],
    'supplementary_figure5': [('supplementary_figure5.py', False),
                              ('fraction_sets.py', False)],
    'supplementary_figure6': [('q1_resolution.py', True),
                              ('supplementary_figure6.py', True)],
    'supplementary_figure7': [('supplementary_figure7.py', True)],
    'supplementary_figure8': [('supplementary_figure8.py', False)],
}

MEASURE = [
    ('Figure 1d: MS2 ion current per cycle', [('figure1', 'measure_ion_current.py', ())]),
    ('Figure 2: retention time x Q1 map and Q1 precision (ZT Scan HDF)',
     [('figure2', 'measure_rt_q1_map.py', ('A9',)),
      ('figure2', 'fit_provisional_q1_axes.py', ()),
      ('figure2', 'measure_q1_precision_wholefile.py', ('--runs', 'A5,A9,B1,B5,B9')),
      ('figure2', 'fit_q1_axis_wholefile.py', ()),
      ('figure2', 'extract_q1_window.py', ('--axis', 'wholefile'))]),
    ('Supplementary Figure 1g, h: mass accuracy and resolution',
     [('supplementary_figure1', 'measure_mass_accuracy_resolution.py', ('8600',)),
      ('supplementary_figure1', 'measure_mass_accuracy_resolution.py', ('7600',))]),
    ('Supplementary Figure 5: LC peak width',
     [('supplementary_figure5', 'measure_lc_peak_width.py', ())]),
    ('Figure 6a-d: Promega 6x5 standard (MRMHR)',
     [('figure6', 'measure_6x5.py', ()), ('figure6', 'measure_6x5_ms1.py', ())]),
    ('Figure 6g and Supplementary Figure 7: pS129 occupancy',
     [('figure6', 'occupancy_transition_check.py', ()),
      ('figure6', 'light_site_ions.py', ()),
      ('supplementary_figure7', 'measure_ps129_ztscan.py', ()),
      ('supplementary_figure7', 'ztscan_site_ions.py', ())]),
    ('Supplementary Figure 8: acquired isolation windows',
     [('supplementary_figure8', 'extract_acquired_windows.py', ())]),
]


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': text.strip('\n').splitlines(True)}


def notebook(cells):
    return {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3',
                                        'language': 'python'},
                         'language_info': {'name': 'python'}}}


SETUP = """
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join('..', 'tools')))
import notebook_helpers as nb
"""

PATHS = """
Inputs are found through three environment variables, set before starting
Jupyter or at the top of this cell (`os.environ[...] = ...` before the import):

| variable | default | holds |
|---|---|---|
| `ZT8600_DATA_ROOT` | `./data` | the data package `zenotof8600_data` (PEAKS outputs, measured tables, artwork) |
| `ZT8600_RAW_ROOT` | `./raw` | the raw deposit `zenotof8600_raw` (.wiff by figure, ZT Scan HDF in `ztscan_full/`); only for raw measurements |
| `ZT8600_OUTPUT_ROOT` | `./output` | panels, source data, figures, caches |
"""


def title(fig):
    n = fig.replace('supplementary_figure', '')
    return (f'Supplementary Figure {n}' if fig.startswith('supplementary')
            else f'Figure {fig[len("figure"):]}')


def figure_notebook(fig):
    cells = [md(f'# {title(fig)}\n\nBuilds the panels of {title(fig).lower()} '
                f'script by script from the data package, shows them and '
                f'assembles the figure. Each cell can be run on its own; a '
                f'missing input stops only that cell and names the folder it '
                f'is expected in.\n' + PATHS),
             code(SETUP),
             md('## Inputs'),
             code(f"nb.check('{fig}')")]
    for script, letters in ENTRY[fig]:
        cells.append(md(f'## `{script}`' + (
            '\n\nPass panel letters to build only some panels, e.g. '
            f"`nb.run('{fig}', '{script}', 'b')`." if letters else '')))
        cells.append(code(f"nb.run('{fig}', '{script}')\nnb.show('{fig}')"))
    cells += [md('## Assembled figure'), code(f"nb.compose('{fig}')")]
    return notebook(cells)


def check_notebook():
    return notebook([
        md('# Input check\n\nLists, figure by figure, which input files are '
           'present and where each missing one is expected. Figures with '
           'missing inputs are skipped by `tools/run_figures.py`; the others '
           'build normally.\n' + PATHS),
        code(SETUP),
        md('## Data package (what the figure scripts read)'),
        code('nb.check()'),
        md('## Raw deposit as well (only for the raw measurements)'),
        code('nb.check(raw=True)'),
        md('## One figure'),
        code("nb.check('figure2', raw=True)")])


def measure_notebook():
    cells = [md('# Measurements from the raw deposit\n\nRe-measures the tables '
                'of the data package that come from the raw files. Not needed '
                'to build the figures. `nb.reuse_raw()` keeps every raw file '
                'that is loaded, so scripts that read the same file share one '
                'copy (memory permitting; restart the kernel to release it).\n'
                + PATHS),
             code(SETUP + 'nb.reuse_raw()'),
             code("nb.check(raw=True)")]
    for heading, calls in MEASURE:
        cells.append(md(f'## {heading}'))
        for fig, script, args in calls:
            a = ''.join(f", '{x}'" for x in args)
            cells.append(code(f"nb.run('{fig}', '{script}'{a})"))
    cells.append(md(
        '## Supplementary Figure 5a: LC peak width\n\nMeasured with the '
        "library workflow `workflows/lc_peak_width_analysis.py`, which takes its "
        "own options; see section 5 of the README."))
    return notebook(cells)


def main():
    os.makedirs(OUT, exist_ok=True)
    books = {'00_check_inputs.ipynb': check_notebook(),
             'raw_measurements.ipynb': measure_notebook()}
    for fig in ENTRY:
        books[f'{fig}.ipynb'] = figure_notebook(fig)
    for name, nbk in books.items():
        with open(os.path.join(OUT, name), 'w', encoding='utf-8', newline='\n') as fh:
            json.dump(nbk, fh, indent=1, ensure_ascii=False)
            fh.write('\n')
    print(f'{len(books)} notebooks in {OUT}')


if __name__ == '__main__':
    main()
