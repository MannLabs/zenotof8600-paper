"""Raw-file analysis (manual mode) — see REFACTOR_PLAN.md §7."""

from __future__ import annotations

from ..ion_current import tic, tic_per_spectrum
from .extraction import (
    load_ms_data, extract_xic, sum_spectra, merge_peaks, pick_peak, points_in_window, integrate_peak, filter_spectra, peak_from_fragments, match_peaks, window_indices, peak_shape, measure_resolution_ppm, xic_tolerance_ppm, resolve_mz_tol, locate_peak_mz, measure_mz_shift_ppm, XIC_FWHM_FRACTION, XIC_FALLBACK_PPM,
)
from .peaks import profile_tolerance_ok
from .targeted_scans import (
    target_scans,
    assert_targets_resolved,
    targeted_xic,
    is_targeted,
)

from .sampling import stratified_sample
from .profile import (
    centroid_profile,
    lattice_step,
    LATTICE_AT_500,
    GAP_FACTOR,
)

from .metadata import (
    read_acquisition_metadata,
    read_acquisition_time,
    survey_raw_files,
    acquisition_times,
)

from .targeted import (
    EXTRA_FRAGMENT_TYPES, FRAGMENT_TYPES, SIL_MODIFICATIONS, calibration_fit, precursor_isotope_transitions, quantify_targets, register_extra_fragment_types, register_sil_modifications, split_charged_frag_type, summed_fragment_xic, target_fragments, top_n_fragments, y_ions_above_precursor,
)

from . import extraction
from . import metadata
from . import targeted
