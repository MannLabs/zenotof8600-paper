# ZenoTOF 8600 — figure code

Analysis code for the ZenoTOF 8600 manuscript. Every data panel of the six main
and eight supplementary figures is produced by a script in this repository, from
the PEAKS search outputs and the tables measured from the raw files that are
deposited with the paper.

The repository is self-contained: it bundles the analysis library it needs
(`zenotof8600_analytics`), so nothing outside this folder has to be installed
first.

```
8600_code/
├── zt8600_config.py              data, output and raw-file roots
├── pyproject.toml                package metadata (lower bounds)
├── environment.yml               exact versions the figures were produced with
├── requirements.txt              the same pins for a plain venv
├── src/zenotof8600_analytics/    the analysis library
├── workflows/                    raw-file measurement workflows of the library
├── experiments/H032_E3xx/        experiment-specific analysis modules
├── figure1/ … figure6/           panel scripts
├── supplementary_figure1/ … 8/
├── make_main_figures.py          assembles Figure 1–6
├── make_supplementary_composites.py   assembles Supplementary Figure 1–8
├── tools/                        run_figures.py, compare_outputs.py
└── unit_tests/                   tests of the library
```

---

## 1. Install

Python 3.12. Get the code and install it from its folder:

```bash
git clone https://github.com/MannLabs/zenotof8600-paper.git
cd zenotof8600-paper
```


```bash
mamba env create -f environment.yml
conda activate zenotof8600
pip install -e .
```

or, in a plain virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate elsewhere)
pip install -r requirements.txt
pip install -e .
```

Check:

```bash
python -c "import zenotof8600_analytics as core; print(core.__file__)"
python -m pytest -q
```

## 2. Point it at the data

Unpack the deposited data archive into `data/` in this folder, or set the
variables below. `data/` and `output/` are git-ignored.

| variable | default | meaning |
|---|---|---|
| `ZT8600_DATA_ROOT` | `./data` | input package: `<experiment>/<PEAKS search>/`, `<experiment>/<table>`, `artwork/<figure>/` |
| `ZT8600_OUTPUT_ROOT` | `./output` | panels, source data, composite figures, caches |
| `ZT8600_RAW_ROOT` | `./raw` | the raw deposit (`zenotof8600_raw` on MassIVE): `ztscan_full/*.hdf` and the `.wiff` files; only for the measurement scripts, read-only |
| `ZT8600_MS_CACHE` | `<output>/cache/ms_data` | parsed-raw-file caches written by the measurement scripts (~5 GB per run) |

Nothing is written into the data package by the panel scripts.

## 3. Build the figures

First check that the inputs are where the code expects them (`--raw` adds
the raw deposit, which only the measurements of section 5 need):

```bash
python tools/check_inputs.py            # all figures
python tools/check_inputs.py figure2 --raw
```

It lists, per figure, every missing file and the folder it is expected in.
Figures work independently: a figure whose inputs are missing fails on its
own, the others build, and the composites are assembled from whatever
panels exist.

All panels and both composite sets:

```bash
python tools/run_figures.py
```

or one figure (scripts run from their own folder):

```bash
cd figure3
python figure3.py
```

Each panel is written to `output/<figure>/panels/` as PDF (vector, editable
text), PNG (300 dpi) and `_sourcedata.csv` (the plotted values). The composite
figures go to `output/figures/`.

| figure | entry point(s) | runtime |
|---|---|---|
| 1 | `figure1.py` | <1 min |
| 2 | `figure2.py` | ~2 min |
| 3 | `figure3.py` | ~5 min (first run; DirectLFQ results are cached in `output/cache`) |
| 4 | `single_cell.py`, `cell_cycle.py`, `rank_plot.py`, `dvp_tissue.py` | ~2 min |
| 5 | `figure5.py` | ~2 min |
| 6 | `figure6.py` | ~2 min |
| S1–S8 | `supplementary_figureN.py`; S2 also `q1_abundance_occupancy.py`, S5 also `fraction_sets.py`, S6 also `q1_resolution.py` | 0.5–2 min each |

### Notebooks

`notebooks/` holds the same steps for interactive use (`pip install -e
.[notebooks]`): `00_check_inputs.ipynb`, one notebook per figure that builds
its panels script by script, shows them and assembles the figure, and
`raw_measurements.ipynb` for section 5, which loads each raw file once per
session and reuses it across scripts. `tools/make_notebooks.py` regenerates
them.

## 4. Check a rebuild against the published source data

```bash
python tools/compare_outputs.py <folder with figureN/panels/*_sourcedata.csv>
```

compares every rebuilt panel's source data with a reference set: text exactly,
numbers to a relative tolerance of 1e-9.

## 5. From raw files

The tables in the data package that are not PEAKS output were measured from the
raw files. The scripts that measure them sit beside the panels they feed; they
need the `raw` extra (`pip install -e .[raw]`; Windows, SCIEX `.wiff` reader)
and `ZT8600_RAW_ROOT` laid out as `<figure>/<run>.wiff` and `ztscan_full/`.
Parsed raw files are cached under `ZT8600_MS_CACHE`, never in the deposit. The raw deposit `zenotof8600_raw` on MassIVE holds the
`.wiff` files (with `.wiff.scan`, `.wiff2`, `.timeseries.data`) in one folder
per figure, `<figure>/<run>.wiff`, each run under the first figure that uses
it, and the ZT Scan HDF files in `ztscan_full/` (converted with a research-grade API
for ZT Scan files); `RAW_MANIFEST.csv` lists every file. Download it to `./raw` or
point `ZT8600_RAW_ROOT` at it; `zt8600_config.RAW_FIGURE` maps experiments to
their figure folders.

| table in the data package | measured by |
|---|---|
| `figure1/ion_current/*.csv` | `figure1/measure_ion_current.py` |
| `H032_E376/q1_calibration_old/q1_axis_A5.json`, `q1_axis_A9.json` (the provisional Q1 axes the whole-file measurement is centred on) | `figure2/fit_provisional_q1_axes.py`, which runs `experiments/H032_E376/precursor_mass_q1_vs_ms1.py`, `dump_q1_patches.py` and `q1_axis_stability.py` for each; run it before `measure_q1_precision_wholefile.py` |
| `supplementary_figure8/acquired_windows_E352_A1.csv` | `supplementary_figure8/extract_acquired_windows.py` |
| `H032_E376/ztscan_rt_q1_map_A9_grid.npz` | `figure2/measure_rt_q1_map.py A9` |
| `H032_E376/q1_wholefile/*.csv`, `q1_axis_A9_wholefile.json`, `q1_window_A9_wholefile.npz` | `figure2/measure_q1_precision_wholefile.py`, `fit_q1_axis_wholefile.py`, `extract_q1_window.py` |
| `H032_E376/q1_axis_transfer_A9_to_A10.csv` (the A9 axis applied to the second 2.8 Th injection, A10) | `figure2/q1_axis_transfer.py`, after `measure_q1_precision_wholefile.py` has measured A9 and A10 |
| `H032_E352/*.rawdiag.parquet`, `H032_E303/*.rawdiag.parquet` | `supplementary_figure1/measure_mass_accuracy_resolution.py 8600` / `7600` |
| `H032_E350/lc_peak_width_sourcedata.csv` | `supplementary_figure5/measure_lc_peak_width.py` (runs `workflows/lc_peak_width_analysis.py`) |
| `H032_E396/6x5_*`, `ms1/6x5_*` | `figure6/measure_6x5.py`, `measure_6x5_ms1.py` |
| `H032_E397/transition_check/*`, `light_site_ions.csv` | `figure6/occupancy_transition_check.py`, `light_site_ions.py` |
| `H032_E397/ztscan_ps129_occupancy.csv`, `ztscan_site_ions.csv` | `supplementary_figure7/measure_ps129_ztscan.py`, `ztscan_site_ions.py` |
| `H032_E391/summed_spectra/*`, `H032_E396/standard_*.csv`, `H032_E396/trace_table_*.csv`, `H032_E276/q1_resolution_*.csv` | the panel scripts themselves (`figure6/spectra.py`, `standard.py`, `supplementary_figure6/supplementary_figure6.py`, `q1_resolution.py`), when the table is absent |

## License

MIT (see `LICENSE`). Authors: Tim Heymann; Claude (Anthropic).
