"""Plotting subpackage."""

from __future__ import annotations

from ._style import (
    init_plotting, PALETTE, PALETTE_DOUBLE, PALETTE_SINGLE, PALETTE_CV_CATEGORIES,
    DEFAULT_PANEL_SIZE, set_default_panel_size,
    replicate_point_size, drawn_bar_width_inches, scale_replicate_points,
    set_replicate_point_scale, set_axes_size_inches,
)
from .binned import plot_binned_median
from .cv import (
    plot_cv_violin, plot_cv_ecdf,
)
from .bars import (
    plot_bar, plot_boxplot_with_points, plot_overlapping_bars, UNION_TO_PER_RUN,
)
from .pca import plot_pca
from .rank import plot_rank
from .volcano import plot_volcano
from .bars import (
    BAR_PITCH_IN, BAR_WIDTH_IN, BAR_GROUP_GAP_IN, fix_bar_geometry,
)

from . import (
    _style, binned, cv, bars, pca, rank, volcano,
)
