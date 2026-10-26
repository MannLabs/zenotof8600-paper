"""Helpers for the notebooks: run a figure script in this session, show its output."""

import functools
import glob
import importlib.util
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import zt8600_config as cfg

_MODULES = {}


def module(figure, script):
    """The script's module, imported once from its own folder."""
    key = (figure, script)
    if key not in _MODULES:
        folder = os.path.join(REPO, figure)
        if folder not in sys.path:
            sys.path.insert(0, folder)
        name = os.path.splitext(script)[0]
        spec = importlib.util.spec_from_file_location(name, os.path.join(folder, script))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        cwd = os.getcwd()
        os.chdir(folder)
        try:
            spec.loader.exec_module(mod)
        finally:
            os.chdir(cwd)
        _MODULES[key] = mod
    return _MODULES[key]


def run(figure, script, *letters):
    """Build panels: the script's main(), restricted to ``letters`` when it takes them."""
    import inspect
    import matplotlib
    matplotlib.use('Agg')
    mod = module(figure, script)
    if letters and not inspect.signature(mod.main).parameters:
        print(f'{script} builds all of its panels at once; letters ignored')
        letters = ()
    cwd = os.getcwd()
    os.chdir(os.path.join(REPO, figure))
    try:
        return mod.main(*letters)
    finally:
        os.chdir(cwd)
        import matplotlib.pyplot as plt
        plt.close('all')


def show(figure, letters=None, width=420):
    """Display the figure's panels (PNG) built so far, optionally only ``letters``."""
    from IPython.display import Image, Markdown, display
    files = sorted(glob.glob(os.path.join(cfg.output_dir(figure, 'panels'), '*.png')))
    shown = 0
    for f in files:
        name = os.path.basename(f)
        letter = name[len(figure):len(figure) + 1]
        if name.startswith('deleteme') or (letters and letter not in letters):
            continue
        display(Markdown(f'**{name[:-4]}**'), Image(f, width=width))
        shown += 1
    if not shown:
        print(f'no panels yet in {cfg.output_dir(figure, "panels")}')


def compose(figure):
    """Assemble the figure from whatever panels exist and display it."""
    from IPython.display import Image, display
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        if figure.startswith('supplementary'):
            import make_supplementary_composites as sc
            sc.main()
            stem = f'Supplementary_Figure{figure[len("supplementary_figure"):]}'
        else:
            import make_main_figures as mf
            mf.main(figure[len('figure'):])
            stem = f'Figure{figure[len("figure"):]}'
    finally:
        os.chdir(cwd)
    png = os.path.join(cfg.output_dir('figures'), stem + '.png')
    if os.path.exists(png):
        display(Image(png, width=800))


def check(figure=None, raw=False):
    """tools/check_inputs.py for one figure (or all), as a table."""
    sys.path.insert(0, os.path.join(REPO, 'tools'))
    import check_inputs
    t = check_inputs.check([figure] if figure else None, raw=raw)
    check_inputs.report(t)
    return check_inputs.summary(t)


def reuse_raw():
    """Load each raw file once per session, whichever script asks for it."""
    import zenotof8600_analytics as core
    raw = core.raw
    if getattr(raw.load_ms_data, '_memoised', False):
        return
    original = raw.load_ms_data

    @functools.wraps(original)
    def load_once(path, *args, **kwargs):
        key = (str(path), args, tuple(sorted((k, repr(v)) for k, v in kwargs.items())))
        if key not in load_once.cache:
            load_once.cache[key] = original(path, *args, **kwargs)
        return load_once.cache[key]

    load_once.cache = {}
    load_once._memoised = True
    raw.load_ms_data = load_once
    for name, mod in list(sys.modules.items()):
        if name.startswith('zenotof8600_analytics') and getattr(mod, 'load_ms_data', None) is original:
            mod.load_ms_data = load_once
    print('raw files are now loaded once per session')
