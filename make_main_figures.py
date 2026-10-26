"""Assemble the six main figures from their panels -- replaces hand assembly."""

import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject

import make_supplementary_composites as sc

HERE = os.path.dirname(os.path.abspath(__file__))
import zt8600_config as cfg

PAGE_W_IN = sc.MAX_PAGE_W_IN
MAX_PAGE_H_IN = sc.MAX_PAGE_H_IN
MARGIN_IN = 0.06
CONTENT_W_IN = PAGE_W_IN - 2 * MARGIN_IN
MIN_GAP_X_IN = 0.06
GAP_Y_IN = 0.04
LETTER_BAND_IN = 0.20
STACK_GAP_IN = 0.04
LEGEND_PT, AXIS_PT, FLOOR_PT, COLUMN_MM = 10.0, 10.0, 8.0, 180.0

LAYOUT = {
    1: [['a'], ['b', 'c', 'd']],
    2: [['a', 'b', 'c'], ['d', 'e'], ['f', 'g']],
    3: [['a', 'b', 'c', 'd'], ['e'], ['f', 'g', 'h']],
    4: [['a'], ['b', 'c', 'd'], ['e', 'f', 'g']],
    5: [['a', 'b'], ['c', 'd', 'e']],
    6: [['a', 'b', ('c', 'd')], ['e', 'f', 'g'], ['h'], ['i']],
}
ARTWORK = {1: {'a'}, 4: {'a', 'e'}, 5: {'a'}, 6: {'e'}}
CONNECTORS = {
    2: [('d', 'zoom_top_right', 'e', 'zoom_top_left'),
        ('d', 'zoom_bottom_right', 'e', 'zoom_bottom_left')],
}
CONNECTOR_STYLE = dict(color='#555555', lw=0.8, ls=(0, (4, 3)))


def panel_files(n):
    """{letter: path}; on a duplicate letter the NEWEST file, with a warning."""
    out = {}
    for f in (glob.glob(os.path.join(cfg.output_dir(f'figure{n}', 'panels'), '*.pdf'))
              + glob.glob(cfg.artwork(f'figure{n}', '*.pdf'))):
        b = os.path.basename(f)
        if b.lower().startswith('deleteme'):
            continue
        m = re.match(rf'[Ff]igure{n}([a-z])[_.]', b)
        if m:
            out.setdefault(m.group(1), []).append(f)
    for k, v in out.items():
        v.sort(key=os.path.getmtime)
        if len(v) > 1:
            print(f'  !! figure {n} {k}: {len(v)} files, using '
                  f'{os.path.basename(v[-1])}')
    return {k: v[-1] for k, v in out.items()}


def layout(n, max_scale=1.0):
    files = panel_files(n)
    art = ARTWORK.get(n, set())
    rows = LAYOUT[n]
    letters = [l for row in rows for s in row
               for l in (s if isinstance(s, tuple) else (s,))]
    missing = [l for l in letters if l not in files]
    if missing:
        raise SystemExit(f'figure {n}: no panel file for {missing} in '
                         f'{cfg.output_dir(f"figure{n}", "panels")} - run the '
                         f'figure{n} scripts first')
    extra = sorted(set(files) - set(letters))
    if extra:
        print(f'  note: figure {n} panels not in LAYOUT: {extra}')
    size = {l: sc.size_in(files[l]) for l in letters}

    def slot_wh(s, k=1.0):
        mem = s if isinstance(s, tuple) else (s,)
        w = max(size[l][0] for l in mem) * k
        h = sum(size[l][1] for l in mem) * k + STACK_GAP_IN * (len(mem) - 1)
        return w, h

    scale = max_scale
    for row in rows:
        data = [s for s in row if s not in art]
        if not data:
            continue
        need = sum(slot_wh(s)[0] for s in data)
        room = CONTENT_W_IN - MIN_GAP_X_IN * (len(row) - 1)
        if len(row) > len(data):
            room -= 1.5
        scale = min(scale, room / need)

    placed, y = [], MARGIN_IN
    for row in rows:
        letter_y = y
        y += LETTER_BAND_IN
        data = [s for s in row if s not in art]
        data_w = sum(slot_wh(s, scale)[0] for s in data)
        data_h = max((slot_wh(s, scale)[1] for s in data), default=0.0)
        arts = [s for s in row if s in art]
        free = CONTENT_W_IN - data_w - MIN_GAP_X_IN * (len(row) - 1)
        geo = {}
        for s in arts:
            aw, ah = size[s]
            k = (free / len(arts)) / aw
            if data_h and ah * k > data_h:
                k = data_h / ah
            geo[s] = (aw * k, ah * k)
        for s in data:
            geo[s] = slot_wh(s, scale)
        widths = [geo[s][0] for s in row]
        gap = ((CONTENT_W_IN - sum(widths)) / (len(row) - 1)
               if len(row) > 1 else 0.0)
        def first(s):
            return (s if isinstance(s, tuple) else (s,))[0]
        off = {s: ((sc.top_offset(files[first(s)]) or 0.0) * scale
                   if s not in art else 0.0) for s in row}
        line = y + max(off.values())
        x, bottom = MARGIN_IN, y
        for s in row:
            mem = s if isinstance(s, tuple) else (s,)
            yy = line - off[s]
            for i, l in enumerate(mem):
                if s in art:
                    w, h = geo[s]
                else:
                    w, h = size[l][0] * scale, size[l][1] * scale
                ly = (letter_y if i == 0
                      else yy - LETTER_BAND_IN + sc.LETTER_INSET_IN)
                placed.append((l, files[l], x, yy, w, h, ly))
                bottom = max(bottom, yy + h)
                yy += h + STACK_GAP_IN
            x += geo[s][0] + gap
        y = bottom + GAP_Y_IN
    page_h = y - GAP_Y_IN + MARGIN_IN
    return scale, (PAGE_W_IN, page_h), placed


def fitted_layout(n):
    """`layout`, shrunk until the page fits A4 -- never a refusal."""
    cap = 1.0
    scale, page, placed = layout(n, cap)
    while page[1] > MAX_PAGE_H_IN + 1e-3 and cap > 0.5:
        cap = scale * 0.99
        scale, page, placed = layout(n, cap)
    if cap < 1.0:
        print(f'  note: figure {n} shrunk to data scale {scale:.3f} to fit '
              f'A4 height')
    return scale, page, placed


def full_row_width(n, letter):
    """Saved width (in) at which `letter`, alone in its row, spans the row."""
    files = panel_files(n)
    art = ARTWORK.get(n, set())
    scale = 1.0
    for row in LAYOUT[n]:
        flat = [l for s in row for l in (s if isinstance(s, tuple) else (s,))]
        data = [s for s in row if s not in art]
        if letter in flat or not data:
            continue
        need = sum(max(sc.size_in(files[l])[0]
                       for l in (s if isinstance(s, tuple) else (s,)))
                   for s in data)
        room = CONTENT_W_IN - MIN_GAP_X_IN * (len(row) - 1)
        if len(row) > len(data):
            room -= 1.5
        scale = min(scale, room / need)
    return CONTENT_W_IN / scale


def connector_lines(n, placed):
    """Page coordinates (inches, from top-left) for each connector."""
    where = {l: (x, y, w, h, p) for l, p, x, y, w, h, _ly in placed}
    lines = []
    for a, ka, b, kb in CONNECTORS.get(n, []):
        pts = []
        for l, key in ((a, ka), (b, kb)):
            x, y, w, h, path = where[l]
            js = path[:-4] + '.anchors.json'
            anc = (json.load(open(js, encoding='utf-8'))
                   if os.path.exists(js) else None)
            pw, ph = sc.size_in(path)
            if (anc is None or key not in anc.get('points', {})
                    or abs(anc['crop_w_in'] - pw) > 0.02
                    or abs(anc['crop_h_in'] - ph) > 0.02):
                print(f'  note: figure {n}: no current anchor {key!r} for '
                      f'panel {l} -- connector skipped')
                pts = None
                break
            k = w / pw
            ax_, ay_ = anc['points'][key]
            pts.append((x + ax_ * k, y + h - ay_ * k))
        if pts:
            lines.append(pts)
    return lines


def overlay_page(placed, page, lines):
    """Letters (Arial Bold 16, as the supplementary pages) + connectors."""
    pw, ph = page
    plt.rcParams['pdf.fonttype'] = 42
    fig = plt.figure(figsize=(pw, ph))
    for (x0, y0), (x1, y1) in lines:
        fig.add_artist(plt.Line2D([x0 / pw, x1 / pw],
                                  [1 - y0 / ph, 1 - y1 / ph],
                                  transform=fig.transFigure,
                                  **CONNECTOR_STYLE))
    for letter, _p, x, _y, _w, _h, ly in placed:
        fig.text(x / pw, (ph - ly - 0.02) / ph, letter,
                 fontname=sc.LETTER_FONT, fontweight='bold',
                 fontsize=sc.LETTER_PT, color='black', ha='left', va='top')
    tmp = os.path.join(cfg.output_dir('figures'), '_overlay_tmp.pdf')
    fig.savefig(tmp, transparent=True)
    plt.close(fig)
    with open(tmp, 'rb') as fh:
        data = fh.read()
    os.unlink(tmp)
    import io
    return PdfReader(io.BytesIO(data)).pages[0]


def write_pdf(placed, page, lines, out):
    pw, ph = page
    w = PdfWriter()
    pg = w.add_blank_page(width=pw * 72, height=ph * 72)
    for _l, path, x, y, wi, hi, _ly in placed:
        src = PdfReader(path).pages[0]
        box = src.mediabox
        s = wi / (float(box.width) / 72)
        op = (Transformation().translate(-float(box.left), -float(box.bottom))
              .scale(s).translate(x * 72, (ph - y - hi) * 72))
        pg.merge_transformed_page(src, op)
    pg.mediabox = RectangleObject((0, 0, pw * 72, ph * 72))
    pg.merge_page(overlay_page(placed, page, lines))
    with open(out, 'wb') as fh:
        w.write(fh)


def write_png(placed, page, lines, out, dpi=200):
    px = lambda v: int(round(v * dpi))
    im = Image.new('RGB', (px(page[0]), px(page[1])), 'white')
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype('arialbd.ttf', int(round(sc.LETTER_PT / 72 * dpi)))
    small = ImageFont.truetype('arial.ttf', int(round(10 / 72 * dpi)))
    for l, path, x, y, w, h, ly in placed:
        png = path[:-4] + '.png'
        if os.path.exists(png):
            im.paste(Image.open(png).convert('RGB').resize((px(w), px(h)),
                                                           Image.LANCZOS),
                     (px(x), px(y)))
        else:
            dr.rectangle([px(x), px(y), px(x + w), px(y + h)],
                         outline='#999999', width=2, fill='#F4F4F4')
            dr.text((px(x + w / 2), px(y + h / 2)),
                    f'{os.path.basename(path)}\n(no PNG preview)',
                    fill='#666666', font=small, anchor='mm', align='center')
        dr.text((px(x), px(ly + 0.02)), l, font=font, fill='black',
                anchor='lt')
    for (x0, y0), (x1, y1) in lines:
        import math
        L = math.hypot(x1 - x0, y1 - y0)
        n = max(int(L / 0.07), 1)
        for i in range(0, n, 2):
            a, b = i / n, min((i + 1) / n, 1)
            dr.line([px(x0 + (x1 - x0) * a), px(y0 + (y1 - y0) * a),
                     px(x0 + (x1 - x0) * b), px(y0 + (y1 - y0) * b)],
                    fill='#555555', width=2)
    im.save(out, dpi=(dpi, dpi))


def main(*which):
    figs = [int(n) for n in which] or sorted(LAYOUT)
    skipped = []
    for n in figs:
        try:
            scale, page, placed = fitted_layout(n)
        except SystemExit as e:
            print(f'skipped {e}')
            skipped.append(n)
            continue
        lines = connector_lines(n, placed)
        stem = os.path.join(cfg.output_dir('figures'), f'Figure{n}')
        write_pdf(placed, page, lines, stem + '.pdf')
        if not sc.render_png(stem + '.pdf', stem + '.png', dpi=300):
            write_png(placed, page, lines, stem + '.png', dpi=300)
        k = scale * (COLUMN_MM / 25.4) / page[0]
        ax_pt, lg_pt = AXIS_PT * k, LEGEND_PT * k
        flag = '' if lg_pt >= FLOOR_PT - 0.05 else \
            f'  << legend under the {FLOOR_PT:g} pt floor'
        print(f'figure {n}: data scale {scale:.3f}, page {page[0]:.2f} x '
              f'{page[1]:.2f} in, prints axis {ax_pt:.1f} / legend '
              f'{lg_pt:.1f} pt at {COLUMN_MM:.0f} mm{flag}'
              + (f', {len(lines)} connector(s)' if lines else ''))
    if skipped:
        print(f'{len(skipped)} figure(s) not composed: {skipped}')


if __name__ == '__main__':
    main(*sys.argv[1:])
