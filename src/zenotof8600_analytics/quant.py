"""DirectLFQ protein-group quantification wrapper (engine-agnostic)."""

from __future__ import annotations

import os
import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd


def compute_directlfq_pg_intensity(
    df: pd.DataFrame,
    *,
    protein_col: str = 'protein_group',
    ion_col: str = 'precursor_id',
    intensity_col: str = 'precursor_intensity',
    run_col: str = 'run',
    num_cores: int = 8,
    min_nonan: int = 1,
    quiet: bool = True,
    use_inmemory: bool = True,
) -> pd.Series:
    """Compute a protein-group-level intensity per (protein_group, run) via DirectLFQ, and return a Series aligned to `df.index` with the broadcast `pg_intensity` value for every input row.

    Parameters:
      use_inmemory: when True (default), invoke DirectLFQ's internal pipeline
                    on the in-memory wide DataFrame, skipping the TSV write/
                    read round-trip on both sides. Falls back automatically to
                    the file-based path if any DirectLFQ internal symbol is
                    missing (e.g. on a future version that renames them).
                    Pass False to force the file-based path.
    """
    needed = [protein_col, ion_col, intensity_col, run_col]
    sub = df[needed]
    sub = sub[sub[intensity_col].notna() & (sub[intensity_col] > 0)]

    wide = sub.pivot_table(
        index=[ion_col, protein_col],
        columns=run_col,
        values=intensity_col,
        aggfunc='first',
    ).reset_index()
    wide = wide.rename(columns={protein_col: 'protein', ion_col: 'ion'})

    if use_inmemory:
        try:
            prot_wide = _run_directlfq_inmemory(
                wide, num_cores=num_cores, min_nonan=min_nonan, quiet=quiet,
            )
        except (AttributeError, ImportError, TypeError) as e:
            warnings.warn(
                f'DirectLFQ in-memory path failed ({type(e).__name__}: {e}); '
                f'falling back to file-based path. Pass use_inmemory=False to '
                f'silence this.'
            )
            prot_wide = _run_directlfq_via_files(
                wide, num_cores=num_cores, min_nonan=min_nonan, quiet=quiet,
            )
    else:
        prot_wide = _run_directlfq_via_files(
            wide, num_cores=num_cores, min_nonan=min_nonan, quiet=quiet,
        )

    sample_cols = [c for c in prot_wide.columns if c != 'protein']
    prot_long = prot_wide.melt(
        id_vars=['protein'], value_vars=sample_cols,
        var_name=run_col, value_name='pg_intensity',
    )
    prot_long = prot_long.rename(columns={'protein': protein_col})

    keyed = df[[protein_col, run_col]].copy()
    keyed['_idx'] = np.arange(len(keyed))
    merged = keyed.merge(prot_long, on=[protein_col, run_col], how='left')
    merged = merged.sort_values('_idx')
    return pd.Series(merged['pg_intensity'].values, index=df.index, name='pg_intensity')


def _run_directlfq_inmemory(wide, *, num_cores, min_nonan, quiet):
    """Drive DirectLFQ's pipeline on an in-memory wide DataFrame, skipping both the TSV write (input) and TSV read (output)."""
    import directlfq.config as _config
    import directlfq.normalization as _lfqnorm
    import directlfq.protein_intensity_estimation as _lfqprot
    import directlfq.utils as _lfqutils

    _config.set_global_protein_and_ion_id(protein_id='protein', quant_id='ion')
    _config.set_log_processed_proteins(log_processed_proteins=True)
    _config.set_compile_normalized_ion_table(compile_normalized_ion_table=True)
    _config.check_wether_to_copy_numpy_arrays_derived_from_pandas()

    ctx = _suppress_directlfq_output() if quiet else _NullContext()
    with ctx:
        input_df = wide.drop_duplicates(subset='ion')
        input_df = _lfqutils.sort_input_df_by_protein_and_quant_id(input_df)
        input_df = _lfqutils.remove_potential_quant_id_duplicates(input_df)
        input_df = _lfqutils.index_and_log_transform_input_df(input_df)
        input_df = _lfqutils.remove_allnan_rows_input_df(input_df)
        input_df = _lfqnorm.NormalizationManagerSamplesOnSelectedProteins(
            input_df,
            num_samples_quadratic=50,
            selected_proteins_file=None,
        ).complete_dataframe
        protein_df, _ion_df = _lfqprot.estimate_protein_intensities(
            input_df,
            min_nonan=min_nonan,
            num_samples_quadratic=10,
            num_cores=num_cores,
        )
    return protein_df


def _run_directlfq_via_files(wide, *, num_cores, min_nonan, quiet):
    """Original file-based path: write the wide DataFrame to a temp TSV, call DirectLFQ's `run_lfq`, read the protein TSV back."""
    import directlfq.lfq_manager as lfq_manager
    import pyarrow as pa
    import pyarrow.csv as pacsv

    workdir = tempfile.mkdtemp(prefix='directlfq_')
    try:
        in_path = os.path.join(workdir, 'input.aq_reformat.tsv')
        pacsv.write_csv(
            pa.Table.from_pandas(wide, preserve_index=False),
            in_path,
            write_options=pacsv.WriteOptions(
                include_header=True,
                delimiter='\t',
                quoting_style='none',
            ),
        )
        kwargs = dict(input_file=in_path, num_cores=num_cores, min_nonan=min_nonan)
        if quiet:
            with _suppress_directlfq_output():
                lfq_manager.run_lfq(**kwargs)
        else:
            lfq_manager.run_lfq(**kwargs)
        prot_wide = pd.read_csv(in_path + '.protein_intensities.tsv', sep='\t')
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return prot_wide


class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *exc): return False


class _suppress_directlfq_output:
    """Silence DirectLFQ chatter — both `print` (stdout) and the `directlfq` logger (stderr)."""

    def __enter__(self):
        import logging
        import sys
        self._stdout = sys.stdout
        self._buf = open(os.devnull, 'w')
        sys.stdout = self._buf
        self._logger = logging.getLogger('directlfq')
        self._prev_level = self._logger.level
        self._logger.setLevel(logging.WARNING)
        return self

    def __exit__(self, *exc):
        import sys
        sys.stdout = self._stdout
        self._buf.close()
        self._logger.setLevel(self._prev_level)

