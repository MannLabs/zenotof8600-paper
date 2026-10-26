"""Provenance for partially-built figures — what makes `python figure3.py c` safe."""

import hashlib
import json
import os

MANIFEST = '.build.json'

_SHARED = [os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'figure_style.py')]


def _stat_digest(paths):
    """sha1 over (relative path, size, mtime) of every file under `paths`."""
    items = []
    for p in sorted(paths):
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in sorted(files):
                    fp = os.path.join(root, f)
                    try:
                        st = os.stat(fp)
                    except OSError:
                        continue
                    items.append((os.path.relpath(fp, p), st.st_size,
                                  int(st.st_mtime)))
        elif os.path.exists(p):
            st = os.stat(p)
            items.append((os.path.basename(p), st.st_size, int(st.st_mtime)))
    h = hashlib.sha1()
    for it in items:
        h.update(repr(it).encode())
    return h.hexdigest()[:12], len(items)


def fingerprint(script, inputs):
    """{'data': ..., 'code': ..., 'n_inputs': ...} for the current state."""
    lib = None
    try:
        import zenotof8600_analytics as core
        lib = os.path.dirname(os.path.abspath(core.__file__))
    except Exception:
        pass
    data, n = _stat_digest(inputs)
    code, _ = _stat_digest([script] + _SHARED + ([lib] if lib else []))
    return {'data': data, 'code': code, 'n_inputs': n,
            'inputs': sorted(os.path.abspath(p) for p in inputs)}


def _path(out_dir):
    return os.path.join(out_dir, MANIFEST)


def load(out_dir):
    try:
        with open(_path(out_dir), encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def record(out_dir, name, fp, when):
    """Stamp one panel."""
    if fp is None:
        return
    man = load(out_dir)
    man[name] = dict(fp, built=when)
    os.makedirs(out_dir, exist_ok=True)
    tmp = _path(out_dir) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    os.replace(tmp, _path(out_dir))


def report(out_dir, fp, *, prefix='figure'):
    """Print what on disk no longer matches `fp`."""
    if fp is None:
        return []
    man = load(out_dir)
    present = sorted({os.path.splitext(f)[0].replace('_sourcedata', '')
                      for f in os.listdir(out_dir)
                      if f.startswith(prefix) and f.endswith('.pdf')}
                     ) if os.path.isdir(out_dir) else []

    stale_data, stale_code, untracked, other = [], [], [], []
    for name in present:
        rec = man.get(name)
        if rec is None:
            untracked.append(name)
        elif rec.get('inputs') != fp.get('inputs'):
            other.append(name)
        elif rec.get('data') != fp['data']:
            stale_data.append((name, rec.get('built', '?')))
        elif rec.get('code') != fp['code']:
            stale_code.append((name, rec.get('built', '?')))

    if stale_data:
        print(f'\n  !! {len(stale_data)} panel(s) were built from DIFFERENT '
              f'DATA and are stale:')
        for n, when in stale_data:
            print(f'       {n}   built {when}')
        print('     Rebuild them before assembling the figure.')
    if stale_code:
        print(f'\n  note: {len(stale_code)} panel(s) predate the current code '
              f'(same data):')
        for n, when in stale_code:
            print(f'       {n}   built {when}')
    if untracked:
        print(f'\n  note: {len(untracked)} panel(s) carry no build record '
              f'(made before this was tracked): ' + ', '.join(untracked))
    mine = len(present) - len(other)
    if other:
        print(f'\n  ({len(other)} panel(s) here were built from other inputs '
              f'-- another module of this figure -- and are not checked by '
              f'this run)')
    if not (stale_data or stale_code or untracked):
        print(f'\n  all {mine} panel(s) built from these inputs match the '
              f'current data and code.')
    return [n for n, _ in stale_data]
