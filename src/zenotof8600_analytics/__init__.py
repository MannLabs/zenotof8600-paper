"""zenotof8600_analytics — engine-agnostic MS-proteomics analysis library."""

from __future__ import annotations

from .schema import (
    LONG_DF_REQUIRED, LONG_DF_OPTIONAL, SAMPLE_INFO_REQUIRED,
    SAMPLE_INFO_OPTIONAL, ALPHABASE_MODS, validate_df, validate_sample_info,
)
from .proteins import (
    group_proteins_by_shared_peptides, group_proteins_by_signature,
    build_peptide_id,
)
from .sequences import (
    count_missed_cleavages,
)
from .stats import (
    _compute_cv_table, binned_stats, sigma_iqr,
)

from .filters import (
    filter_runs, filter_outlier_runs,
)
from .quant import compute_directlfq_pg_intensity
from .pipeline import process_experiment
from .species import (
    SPECIES_SUFFIXES,
    assign_species,
    expected_log2_ratios,
    plot_expected_composition,
    plot_ratio_accuracy,
    plot_ratio_precision,
    plot_species_counts,
    plot_species_cv_ecdf,
    plot_species_ratio,
    species_ratio_accuracy,
    species_ratio_precision,
    species_ratio_table,
    sum_precursors_to_protein,
)

from .plotting import *

from .io import (
    load_experiments, detect_engine, sample_info_from_experiments, load_peaks, load_diann, pick_output_file, PEAKS_CANDIDATES, DIANN_REJECT,
)

from .plate_map import (
    PLATE_FIELDS, parse_run_position, plate_metadata,
)
from .ion_current import tic, tic_per_spectrum
from .run_metrics import (
    auto_rt_window, seed_rt_window, MIN_SCANS_IN_PROBE, METRICS,
)

from .dia_method import (
    WindowScheme, calculate_forbidden_zones, design_windows, evaluate_windows, find_closest_forbidden_zone, load_precursor_mz, window_occupancy,
)

from .targeted_method import (
    CE_RULES, Q1_ISOTOPE_CHOICES, SCIEX_MRMHR_FORMATS, SCIEX_MRMHR_LAYOUTS,
    precursor_q1_mz, sciex_mrmhr_table, write_sciex_mrmhr_table,
)


from . import (
    schema, proteins, sequences, stats, filters, quant, pipeline, plotting, io, plate_map, species, dia_method,
)

__version__ = "0.1.0"


def __getattr__(name):
    """Lazily expose the optional `raw` subpackage as ``core.raw``."""
    if name == 'raw':
        import importlib
        mod = importlib.import_module(f'{__name__}.raw')
        globals()['raw'] = mod
        return mod
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
