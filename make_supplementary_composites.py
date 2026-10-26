"""Assemble each supplementary figure's panels into one lettered page."""

import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject

HERE = os.path.dirname(os.path.abspath(__file__))
import zt8600_config as cfg

MAX_PAGE_W_IN, MAX_PAGE_H_IN = 8.268, 11.693
MARGIN_IN = 0.088
GAP_X_IN, GAP_Y_IN = 0.10, 0.14

LETTER_PT = 16.0
LETTER_FONT = 'Arial'
LETTER_INSET_IN = 0.10
LETTER_BAND_IN = 0.20

ROWS = {
    'supplementary_figure1': [('a', 'b', 'c'), ('d', 'e', 'f'), ('g', 'h')],
    'supplementary_figure2': [('a', 'b'), ('c', 'd'), ('e',)],
    'supplementary_figure3': [('a',), ('b', 'c', 'd')],
    'supplementary_figure4': [('a', 'b'), ('c', 'd', 'e'), ('f', 'g')],
    'supplementary_figure5': [('a', 'b'), ('c', 'd'), ('e',)],
    'supplementary_figure6': [('a', 'b', 'c'), ('d', 'e'), ('f', 'g', 'h')],
    'supplementary_figure7': [('a', 'b'), ('c',), ('d',)],
    'supplementary_figure8': [('a',), ('b',)],
}

AXIS_PT, FLOOR_PT, COLUMN_MM = 10.0, 8.0, 180.0

CONTENT_W_IN = MAX_PAGE_W_IN - 2 * MARGIN_IN


def panels(folder):
    """{letter: path}, live panels only, keyed by the letter in the filename."""
    out = {}
    for f in sorted(glob.glob(os.path.join(cfg.output_dir(folder, 'panels'), '*.pdf'))):
        name = os.path.basename(f)
        if name.startswith('deleteme_'):
            continue
        m = re.match(r'supplementary_figure\d+([a-z])_', name)
        if m:
            out.setdefault(m.group(1), []).append(f)
    for letter, fs_ in out.items():
        fs_.sort(key=os.path.getmtime)
        if len(fs_) > 1:
            print(f'  !! {folder} {letter}: {len(fs_)} live files, using '
                  f'{os.path.basename(fs_[-1])}; tag the rest deleteme_: '
                  + ', '.join(os.path.basename(x) for x in fs_[:-1]))
    return {k: v[-1] for k, v in out.items()}


def size_in(pdf_path):
    """(width, height) of a panel in inches, from its MediaBox."""
    box = PdfReader(pdf_path).pages[0].mediabox
    return float(box.width) / 72.0, float(box.height) / 72.0


def top_offset(pdf_path):
    """Inches from the panel's top edge down to its topmost DATA area."""
    import json
    js = pdf_path[:-4] + '.anchors.json'
    if not os.path.exists(js):
        return None
    rec = json.load(open(js, encoding='utf-8'))
    if not rec.get('axes'):
        return None
    _w, h = size_in(pdf_path)
    if abs(rec['crop_h_in'] - h) > 0.02:
        print(f'  note: {os.path.basename(js)} does not match its PDF '
              f'({rec["crop_h_in"]:.2f} vs {h:.2f} in tall) -- box-aligned')
        return None
    return h - max(a[3] for a in rec['axes'])


def layout(folder):
    """Where every panel goes, in inches from the page's top-left."""
    have = panels(folder)
    rows = [r for r in ([l for l in row if l in have] for row in ROWS[folder]) if r]
    missing = set(have) - {l for row in rows for l in row}
    if missing:
        print(f'  !! {folder}: panels {sorted(missing)} are not in ROWS and '
              f'are left out -- delete the stale files')

    sizes = {l: size_in(have[l]) for l in have}
    scale = min(1.0, min((CONTENT_W_IN - GAP_X_IN * (len(row) - 1))
                         / sum(sizes[l][0] for l in row) for row in rows))

    def place(scale):
        placed, y = [], MARGIN_IN
        for row in rows:
            letter_y = y
            y += LETTER_BAND_IN
            off = {l: (top_offset(have[l]) or 0.0) * scale for l in row}
            line = y + max(off.values())
            x, bottom = MARGIN_IN, y
            for l in row:
                w, h = (v * scale for v in sizes[l])
                yy = line - off[l]
                placed.append((l, have[l], x, yy, w, h, letter_y))
                x += w + GAP_X_IN
                bottom = max(bottom, yy + h)
            y = bottom + GAP_Y_IN
        page = (max(x + w for _l, _p, x, _y, w, _h, _ly in placed) + MARGIN_IN,
                max(y + h for _l, _p, _x, y, _w, h, _ly in placed) + MARGIN_IN)
        return placed, page

    placed, page = place(scale)
    while page[1] > MAX_PAGE_H_IN + 1e-3 and scale > 0.5:
        scale *= 0.99
        placed, page = place(scale)
    if page[0] > MAX_PAGE_W_IN + 1e-3:
        raise ValueError(f'{folder}: layout is {page[0]:.2f} in wide, past '
                         f'A4 -- regroup ROWS')
    return scale, page, placed


def full_row_width(folder, letter):
    """Saved width (in) at which `letter`, alone in its row, matches the page."""
    have = panels(folder)
    rows = [[l for l in row if l in have] for row in ROWS[folder]
            if letter not in row]
    rows = [r for r in rows if r]
    w = {l: size_in(have[l])[0] for r in rows for l in r}
    scale = min(1.0, min((CONTENT_W_IN - GAP_X_IN * (len(r) - 1))
                         / sum(w[l] for l in r) for r in rows))
    return max(sum(w[l] for l in r) + GAP_X_IN * (len(r) - 1) / scale
               for r in rows)


def write_pdf(folder, placed, page_size, out_path):
    """The composite itself: panels merged onto a blank A4, letters on top."""
    page_w, page_h = page_size
    writer = PdfWriter()
    page = writer.add_blank_page(width=page_w * 72, height=page_h * 72)

    for letter, path, x, y_top, w, h, _ly in placed:
        src = PdfReader(path).pages[0]
        box = src.mediabox
        s = w / (float(box.width) / 72.0)
        ty = (page_h - y_top - h) * 72
        op = (Transformation()
              .translate(-float(box.left), -float(box.bottom))
              .scale(s)
              .translate(x * 72, ty))
        page.merge_transformed_page(src, op)

    page.mediabox = RectangleObject((0, 0, page_w * 72, page_h * 72))
    page.merge_page(_letters_page(placed, page_size))
    with open(out_path, 'wb') as fh:
        writer.write(fh)


def _letters_page(placed, page_size):
    """A transparent A4 page carrying only the panel letters."""
    page_w, page_h = page_size
    plt.rcParams['pdf.fonttype'] = 42
    fig = plt.figure(figsize=(page_w, page_h))
    for letter, _path, x, _y_top, _w, _h, ly in placed:
        fig.text(x / page_w,
                 (page_h - ly - 0.02) / page_h,
                 letter, fontname=LETTER_FONT, fontweight='bold',
                 fontsize=LETTER_PT, color='black',
                 ha='left', va='top')
    tmp = os.path.join(cfg.output_dir('figures'), '_letters_tmp.pdf')
    fig.savefig(tmp, transparent=True)
    plt.close(fig)
    page = PdfReader(tmp).pages[0]
    os.remove(tmp)
    return page


def render_png(pdf_path, out_path, dpi=300):
    """The preview, rendered FROM the finished PDF (pypdfium2) at `dpi`."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return False
    doc = pdfium.PdfDocument(pdf_path)
    img = doc[0].render(scale=dpi / 72.0).to_pil().convert('RGB')
    doc.close()
    img.save(out_path, dpi=(dpi, dpi))
    return True


def write_png(folder, placed, page_size, out_path, dpi=300):
    """A flat preview at 300 dpi, from the panels' own PNGs (fallback)."""
    px = lambda v: int(round(v * dpi))
    canvas = Image.new('RGB', (px(page_size[0]), px(page_size[1])), 'white')
    font = ImageFont.truetype('arialbd.ttf', int(round(LETTER_PT / 72 * dpi)))
    draw = ImageDraw.Draw(canvas)
    for letter, path, x, y_top, w, h, ly in placed:
        png = path[:-4] + '.png'
        if os.path.exists(png):
            im = Image.open(png).convert('RGB')
            canvas.paste(im.resize((px(w), px(h)), Image.LANCZOS),
                         (px(x), px(y_top)))
        draw.text((px(x), px(ly + 0.02)), letter,
                  font=font, fill='black', anchor='la')
    canvas.save(out_path, dpi=(dpi, dpi))


def main():
    for folder in sorted(ROWS):
        have = panels(folder)
        absent = [l for row in ROWS[folder] for l in row if l not in have]
        if len(absent) == sum(len(r) for r in ROWS[folder]):
            print(f'skipped {folder}: no panels in '
                  f'{cfg.output_dir(folder, "panels")} - run its scripts first')
            continue
        if absent:
            print(f'  {folder}: panels {absent} missing - composed without them')
        scale, page_size, placed = layout(folder)
        n = folder.replace('supplementary_figure', '')
        stem = os.path.join(cfg.output_dir('figures'), f'Supplementary_Figure{n}')
        write_pdf(folder, placed, page_size, stem + '.pdf')
        if not render_png(stem + '.pdf', stem + '.png'):
            write_png(folder, placed, page_size, stem + '.png')

        printed = AXIS_PT * scale * (COLUMN_MM / 25.4) / page_size[0]
        flag = '' if printed >= FLOOR_PT else f'  << UNDER THE {FLOOR_PT:g} pt FLOOR'
        print(f'{folder}: {len(placed)} panels, scale {scale:.3f}, '
              f'page {page_size[0]:.2f} x {page_size[1]:.2f} in, '
              f'axis text prints {printed:.1f} pt at {COLUMN_MM:.0f} mm{flag}')
        print(f'  wrote {stem}.pdf|.png')


if __name__ == '__main__':
    main()
