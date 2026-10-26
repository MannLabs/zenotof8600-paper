"""Paths for the ZenoTOF 8600 figure code."""
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get('ZT8600_DATA_ROOT', os.path.join(REPO_ROOT, 'data'))
OUTPUT_ROOT = os.environ.get('ZT8600_OUTPUT_ROOT',
                             os.path.join(REPO_ROOT, 'output'))
RAW_ROOT = os.environ.get('ZT8600_RAW_ROOT', os.path.join(REPO_ROOT, 'raw'))
MS_CACHE_ROOT = os.environ.get('ZT8600_MS_CACHE',
                               os.path.join(OUTPUT_ROOT, 'cache', 'ms_data'))


def figure_of(script_path):
    """Figure folder name of a script, e.g. 'figure2'."""
    return os.path.basename(os.path.dirname(os.path.abspath(script_path)))


def data(*parts):
    """Path inside the input data package."""
    return os.path.join(DATA_ROOT, *parts)


RAW_FIGURE = {
    'H032_E303': 'figure1', 'H032_E352': 'figure1',
    'H032_E376': 'figure2', 'H032_E377': 'figure2',
    'H032_E378': 'figure3',
    'H032_E388': 'figure4', 'H032_E389': 'figure4',
    'H032_E386': 'figure5', 'H032_E392': 'figure5',
    'H032_E391': 'figure6', 'H032_E395': 'figure6', 'H032_E396': 'figure6',
    'H032_E397': 'figure6',
    'H032_E382': 'supplementary_figure4', 'H032_E387': 'supplementary_figure4',
    'H032_E350': 'supplementary_figure5',
    'H032_E276': 'supplementary_figure6',
    'H032_E293': 'supplementary_figure8',
}


def raw(*parts):
    """Path inside the raw deposit."""
    if parts and parts[0] == 'ms_data_cache':
        return os.path.join(MS_CACHE_ROOT, *parts[1:])
    if len(parts) >= 1 and parts[0] in RAW_FIGURE and parts[1:2] in ((), ('raw',)):
        return os.path.join(RAW_ROOT, RAW_FIGURE[parts[0]], *parts[2:])
    return os.path.join(RAW_ROOT, *parts)


def require(path, what='input'):
    """``path`` if it exists, else FileNotFoundError naming the expected folder."""
    if os.path.exists(path):
        return path
    folder, name = os.path.split(path)
    hint = ('the raw deposit (ZT8600_RAW_ROOT)' if path.startswith(RAW_ROOT)
            else 'the data package (ZT8600_DATA_ROOT)'
            if path.startswith(DATA_ROOT) else 'its folder')
    raise FileNotFoundError(f'{what} not found: {name}\n  expected in {folder}\n'
                            f'  part of {hint}; run tools/check_inputs.py for '
                            f'everything that is missing')


def output_dir(figure, *parts):
    """Output directory of a figure, created on demand."""
    d = os.path.join(OUTPUT_ROOT, figure, *parts)
    os.makedirs(d, exist_ok=True)
    return d


def panels(script_path):
    """Panel output directory of the figure a script belongs to."""
    return output_dir(figure_of(script_path), 'panels')


def cache(*parts):
    """Directory for intermediate results written by the measurement scripts."""
    d = os.path.join(OUTPUT_ROOT, 'cache', *parts)
    os.makedirs(d, exist_ok=True)
    return d


def artwork(figure, *parts):
    """Hand-drawn panels (schematics), part of the data package."""
    return os.path.join(DATA_ROOT, 'artwork', figure, *parts)


def _extend_sys_path():
    """Make the workflow scripts and experiment modules importable by name."""
    import sys
    extra = [os.path.join(REPO_ROOT, 'workflows')]
    exp = os.path.join(REPO_ROOT, 'experiments')
    if os.path.isdir(exp):
        extra += [os.path.join(exp, d) for d in sorted(os.listdir(exp))
                  if os.path.isdir(os.path.join(exp, d))]
    for p in [REPO_ROOT, *extra]:
        if p not in sys.path:
            sys.path.append(p)


_extend_sys_path()
