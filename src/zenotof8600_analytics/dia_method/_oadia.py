"""Variable-width DIA isolation-window design (vendored from pyDIAid)."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


def create_binning_with_constraints(
    mz_values: List[float],
    mz_range: Tuple[float, float],
    num_bins: int,
    max_width: float,
    max_iterations: int = 5
) -> List[List[float]]:
    """Creates bins for mass spectrometry data with constraints on bin width.

    Args:
        mz_values: List of m/z values to bin
        mz_range: (min_mz, max_mz) range to consider
        num_bins: Target number of bins
        max_width: Maximum allowed bin width
        max_iterations: Maximum number of refinement iterations

    Returns:
        List of [start, end] ranges for each bin

    Raises:
        ValueError: If constraints cannot be satisfied
    """
    if (mz_range[1] - mz_range[0])/num_bins > max_width:
        raise ValueError(
            "Not all constraints can be fulfilled. Either choose a smaller m/z-range, "
            "more bins, or a wider maximal width."
        )

    bins_extended = list()

    bins_temp, oversized_indices = create_initial_bins_and_check_for_overlength(
        mz_values,
        mz_range,
        num_bins,
        max_width
    )

    if len(oversized_indices) == 0:
        bins_extended.append(bins_temp)
    else:
        bins_extended = reduce_oversized_bins(
            mz_range,
            max_width,
            bins_temp,
            oversized_indices,
            bins_extended
        )

    iteration_count = 0
    while iteration_count < max_iterations:
        _log.debug(iteration_count)
        bins_extended = flatten_and_sort_bins(bins_extended)

        new_mz_ranges, new_num_bins = redistribute_bins_general(
            mz_values,
            mz_range,
            num_bins,
            bins_extended,
            max_width
        )

        all_ranges_processed = True
        for i in range(len(new_mz_ranges)):
            bins_temp, oversized_indices = create_initial_bins_and_check_for_overlength(
                mz_values,
                new_mz_ranges[i],
                new_num_bins[i],
                max_width
            )

            if len(oversized_indices) == 0:
                bins_extended.append(bins_temp)
            else:
                all_ranges_processed = False
                bins_extended = reduce_oversized_bins(
                    new_mz_ranges[i],
                    max_width,
                    bins_temp,
                    oversized_indices,
                    bins_extended
                )

        if all_ranges_processed:
            break

        iteration_count += 1

        if iteration_count == max_iterations:
            _log.debug("Warning: Reached maximum number of iterations")

    return flatten_and_sort_bins(bins_extended)


def reduce_oversized_bins(
    mz_range,
    max_width,
    bins_temp,
    oversized_indices,
    bins_extended,
):

    categorized_bins = categorize_and_merge_oversized_bins(
        bins_temp,
        oversized_indices,
        mz_range
    )
    _log.debug("categoraized", categorized_bins)
    new_bins = transfer_oversized_bins_to_bins_with_maximal_widths(
        categorized_bins,
        max_width
    )
    _log.debug("new_bins", new_bins)
    bins_extended.extend(new_bins)
    _log.debug("current bins", bins_extended, len(bins_extended))

    return bins_extended


def redistribute_bins_general(
    mz_values,
    mz_range,
    num_bins,
    bins_extended,
    max_width
):
    new_num_bins = num_bins - len(bins_extended)
    _log.debug("num bins for next iteration", new_num_bins)

    new_mz_ranges = find_gaps(bins_extended, mz_range)
    _log.debug("new mz ranges", new_mz_ranges)

    items_per_range, weights, bins_per_range_rounded = calculate_bins_per_range(
        mz_values,
        new_mz_ranges,
        new_num_bins
    )
    _log.debug("bins_per_range_rounded", bins_per_range_rounded)

    min_num_bins = list()
    for num in range(len(new_mz_ranges)):
        min_num_bins.append(math.ceil((new_mz_ranges[num][1]-new_mz_ranges[num][0])/max_width))
    _log.debug("min num bins", min_num_bins)

    new_bin_num = redistribute_bins(bins_per_range_rounded, min_num_bins)
    _log.debug("new num bins", new_bin_num)

    return new_mz_ranges, new_bin_num


def categorize_and_merge_oversized_bins(
    bins: List[List[float]],
    oversized_indices: List[int],
    mz_range: Tuple[float, float]
) -> Dict[str, List[Dict[str, any]]]:
    """Categorizes oversized bins by their position relative to the mz range midpoint and merges adjacent oversized bins.

    Parameters:
    bins (List[List[float]]): List of [start, end] ranges for each bin
    oversized_indices (List[int]): Indices of bins that exceed max_width
    mz_range (Tuple[float, float]): (min_mz, max_mz) range to consider

    Returns:
    Dict with keys 'left_region' and 'right_region', each containing a list of
    dictionaries with merged bin information
    """
    if not oversized_indices:
        return {'left_region': [], 'right_region': []}

    mz_midpoint = (mz_range[0] + mz_range[1]) / 2

    left_region = []
    right_region = []

    grouped_indices = []
    current_group = [oversized_indices[0]]

    for i in range(1, len(oversized_indices)):
        if oversized_indices[i] == oversized_indices[i-1] + 1:
            current_group.append(oversized_indices[i])
        else:
            grouped_indices.append(current_group)
            current_group = [oversized_indices[i]]
    grouped_indices.append(current_group)

    for group in grouped_indices:
        start_idx = group[0]
        end_idx = group[-1]

        merged_bin = {
            'mz_range': [bins[start_idx][0], bins[end_idx][1]],
            'bin_width': bins[end_idx][1] - bins[start_idx][0],
        }

        bin_center = (merged_bin['mz_range'][0] + merged_bin['mz_range'][1]) / 2

        if bin_center < mz_midpoint:
            left_region.append(merged_bin)
        else:
            right_region.append(merged_bin)

    return {
        'left_region': left_region,
        'right_region': right_region
    }


def transfer_oversized_bins_to_bins_with_maximal_widths(
    categorized_bins: Dict[str, List[Dict[str, any]]],
    max_width: float
) -> Dict[str, Dict[str, any]]:
    """Generates new windows from oversized regions that respect max_width.

    Parameters:
    categorized_bins (Dict): Output from categorize_and_merge_oversized_bins
    max_width (float): Maximum allowed window width

    Returns:
    Dict containing new windows, remaining ranges, and bin counts for each region
    """

    new_bins = list()

    for merged_bin in categorized_bins['left_region']:
        start_mz = merged_bin['mz_range'][0]
        end_mz = merged_bin['mz_range'][1]
        current_mz = start_mz

        while current_mz + max_width < end_mz:
            new_window = [current_mz, current_mz + max_width]
            new_bins.append(new_window)
            current_mz += max_width

    for merged_bin in categorized_bins['right_region']:
        start_mz = merged_bin['mz_range'][0]
        end_mz = merged_bin['mz_range'][1]
        current_mz = end_mz

        while current_mz - max_width > start_mz:
            new_window = [current_mz - max_width, current_mz]
            new_bins.append(new_window)
            current_mz -= max_width

    return sorted(new_bins)


def find_gaps(intervals, mz_range):
    """Find gaps between intervals in a list of [start, end] intervals.

    Args:
        intervals: List of [start, end] intervals

    Returns:
        List of tuples (gap_start, gap_end) representing gaps between intervals
    """
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    sorted_intervals.insert(0, [mz_range[0], mz_range[0]])
    sorted_intervals.append([mz_range[1], mz_range[1]])
    gaps = []


    for i in range(len(sorted_intervals) - 1):
        current_end = sorted_intervals[i][1]
        next_start = sorted_intervals[i + 1][0]

        if current_end < next_start:
            gaps.append((current_end, next_start))

    return gaps


def calculate_bins_per_range(
    mz_values: List[float],
    mz_ranges: List[Tuple[float, float]],
    total_num_bins: int
) -> Tuple[List[int], List[float], List[int]]:
    """Calculate the optimal number of bins for each m/z range based on data density.

    Args:
        mz_values: List of m/z values
        mz_ranges: List of tuples defining (start, end) of each m/z range
        total_num_bins: Total number of bins to distribute

    Returns:
        Tuple containing:
        - List of items per range
        - List of weights for each range
        - List of number of bins per range (rounded to integers)
    """
    items_per_range = []
    for range_start, range_end in mz_ranges:
        filtered_values = [v for v in mz_values if range_start <= v <= range_end]
        items_per_range.append(len(filtered_values))

    total_items = sum(items_per_range)
    weights = [count / total_items for count in items_per_range]

    bins_per_range = [weight * total_num_bins for weight in weights]

    bins_per_range_rounded = np.round(bins_per_range).astype(int)

    total_assigned = sum(bins_per_range_rounded)
    if total_assigned != total_num_bins:
        diff = total_num_bins - total_assigned
        max_weight_idx = weights.index(max(weights))
        bins_per_range_rounded[max_weight_idx] += diff

    return items_per_range, weights, bins_per_range_rounded.tolist()


def redistribute_bins(weighted_bins, min_bins):
    """Redistributes bins while maintaining minimum requirements and total count.

    Args:
        weighted_bins (list): Initial bin distribution based on weights
        min_bins (list): Minimum number of bins required for each range

    Returns:
        list: Adjusted bin distribution meeting minimum requirements
    """
    result = []
    total_original = sum(weighted_bins)
    total_min = sum(min_bins)

    if total_min >= total_original:
        return min_bins

    remaining_bins = total_original - total_min
    result = min_bins.copy()

    weights = []
    for i in range(len(weighted_bins)):
        if weighted_bins[i] > min_bins[i]:
            weights.append(weighted_bins[i])
        else:
            weights.append(0)

    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w/total_weight if w > 0 else 0 for w in weights]

        for i in range(len(result)):
            if weights[i] > 0:
                extra = round(remaining_bins * weights[i])
                result[i] += extra

    return result


def create_initial_bins_and_check_for_overlength(
    mz_values,
    mz_range,
    num_bins,
    max_width
):
    """Creates Parameters: values (List[float]): List of m/z values to bin mz_range (Tuple[float, float]): (min_mz, max_mz) range to consider num_bins (int): Target number of bins max_width (float): Maximum allowed bin width in Da Returns: .....

    Parameters:
        values (List[float]): List of m/z values to bin
        mz_range (Tuple[float, float]): (min_mz, max_mz) range to consider
        num_bins (int): Target number of bins
        max_width (float): Maximum allowed bin width in Da
    Returns:
        .....
    """
    bins_temp = create_bins_with_varibale_length(
        mz_values,
        mz_range,
        num_bins,
    )
    _log.debug("bins", bins_temp)
    oversized_indices = identify_oversized_bins(bins_temp, max_width)
    _log.debug("oversized bin indices", oversized_indices)
    return bins_temp, oversized_indices


def create_bins_with_varibale_length(
    values: List[float],
    mz_range: Tuple[float, float],
    num_bins: int,
) -> List[List[float]]:
    """Creates variable bins for mass spectrometry data that balance even distribution of items with reasonable bin widths.

    Parameters:
        values (List[float]): List of m/z values to bin
        mz_range (Tuple[float, float]): (min_mz, max_mz) range to consider
        num_bins (int): Target number of bins
        min_width (float): Minimum allowed bin width in Da
        max_width (float): Maximum allowed bin width in Da
        merging_for_small_bins (bool): Defines if only max_width contrain or both,
            min_width and max_width, is applied
    Returns:
        List[List[float]]: List of [start, end] ranges for each bin
    """
    filtered_values = sorted([v for v in values if mz_range[0] <= v <= mz_range[1]])
    if not filtered_values:
        return []
    bins = get_initial_equal_bins(filtered_values, num_bins)
    bins[0][0] = mz_range[0]
    bins[-1][1] = mz_range[1]
    return bins


def identify_oversized_bins(bins: List[List[float]], max_width: float) -> List[int]:
    """Identifies bins that exceed the specified maximum width.

    Parameters:
    bins (List[List[float]]): List of [start, end] ranges for each bin
    max_width (float): Maximum allowed bin width in Da
    Returns:
    List[int]: Indices of bins that exceed max_width
    """
    oversized_bins = []
    for i, bin_range in enumerate(bins):
        bin_width = bin_range[1] - bin_range[0]
        if bin_width > max_width:
            oversized_bins.append(i)
    return oversized_bins


def flatten_and_sort_bins(bins):
    """Flattens and sorts a list of bins where some elements might be nested lists of bins.

    Args:
        bins (list): List of [start, end] ranges, some elements might be lists of ranges
    Returns:
        list: Flattened and sorted list of [start, end] ranges
    """
    flattened = []
    for item in bins:
        if len(item) == 2 and isinstance(item[0], (int, float)):
            flattened.append(item)
        else:
            flattened.extend(item)
    return sorted(flattened, key=lambda x: x[0])


"""-----------------------------------------------------------------------------------------"""


def get_initial_equal_bins(
    values: List[float],
    num_splits: int
) -> List[List[float]]:
    """Creates initial bins with equal number of items by splitting the data into roughly equal chunks.

    Parameters:
        values (List[float]): List of m/z values to be binned
        num_splits (int): Number of desired splits/bins

    Returns:
        List[List[float]]: List of [start, end] ranges for each bin
    """
    remainder = len(values) % num_splits
    if remainder:
        values = values + [values[-1]] * (num_splits - remainder)

    splits = np.array_split(np.array(values), num_splits)

    ranges = []
    for i, split in enumerate(splits):
        if i == 0:
            ranges.append([split[0], split[-1]])
        else:
            ranges.append([ranges[-1][1], split[-1]])
    return ranges


def merge_small_bins(bins, min_width):
    """Merge bins that are smaller than min_width using a forward-looking approach.

    Parameters:
    bins (list): List of [start, end] bin edges
    min_width (float): Minimum allowed bin width

    Returns:
    list: New list of [start, end] bin edges after merging
    """
    result = []
    i = 0

    while i < len(bins):
        start = bins[i][0]
        end = bins[i][1]
        width = end - start

        if width >= min_width:
            result.append([start, end])
            i += 1
            continue

        total_width = width
        j = i + 1
        while j < len(bins) and total_width < min_width:
            total_width += bins[j][1] - bins[j][0]
            j += 1

        accumulated_range = bins[j-1][1] - start
        n_new_bins = max(1, int(np.floor(accumulated_range / min_width)))
        new_width = accumulated_range / n_new_bins

        for k in range(n_new_bins):
            new_start = start + k * new_width
            new_end = start + (k + 1) * new_width
            result.append([new_start, new_end])

        i = j

    if len(result) > 1 and result[-1][1] - result[-1][0] < min_width:
        last_bin = result.pop()
        result[-1][1] = last_bin[1]

    _log.debug("Number of bins merged (reason: size below min_width):", len(bins)-len(result))

    return result


def create_variable_bins(
    values: List[float],
    mz_range: Tuple[float, float],
    num_bins: int,
    min_width: float = 2.0,
    max_width: float = 50.0,
) -> List[List[float]]:
    """Creates variable bins for mass spectrometry data that balance even distribution of items with reasonable bin widths.

    Parameters:
        values (List[float]): List of m/z values to bin
        mz_range (Tuple[float, float]): (min_mz, max_mz) range to consider
        num_bins (int): Target number of bins
        min_width (float): Minimum allowed bin width in Da
        max_width (float): Maximum allowed bin width in Da
        merging_for_small_bins (bool): Defines if only max_width contrain or both,
            min_width and max_width, is applied

    Returns:
        List[List[float]]: List of [start, end] ranges for each bin
    """

    bins = create_binning_with_constraints(
        values,
        mz_range,
        num_bins,
        max_width
    )

    final_bins = merge_small_bins(bins, min_width)
    return final_bins


def get_initial_equal_width_bins(
    mz_range: Tuple[float, float],
    num_splits: int
) -> List[List[float]]:
    """Creates initial bins with equal m/z width by splitting the total range into equal segments.

    Parameters:
        mz_range (Tuple[float, float]): (min_mz, max_mz) range to consider
        num_splits (int): Number of desired splits/bins

    Returns:
        List[List[float]]: List of [start, end] ranges for each bin
    """
    min_val = mz_range[0]
    max_val = mz_range[1]
    total_range = max_val - min_val

    bin_width = total_range / num_splits

    ranges = []
    for i in range(num_splits):
        start = min_val + (i * bin_width)
        end = min_val + ((i + 1) * bin_width)

        if i == num_splits - 1:
            end = max_val

        ranges.append([start, end])

    return ranges


def calculate_forbidden_zones(nominal_mass, phospho_enriched):
    """Calculate forbidden zones for peptide m/z values based on mass defects.

    Parameters:
    -----------
    nominal_mass : float
        The nominal m/z value to calculate the forbidden zone for
    charge_state : int, optional
        Charge state of the peptides (default: 2)
    phospho_enriched : bool, optional
        Whether the sample is phospho-enriched (default: False)

    Returns:
    --------
    float
        The m/z value of the forbidden zone
    """
    optimal_mz_increment = 1.00045475
    optimal_mz_constant = 0.18 if phospho_enriched else 0.25

    return (np.ceil(nominal_mass / optimal_mz_increment) *
            optimal_mz_increment + optimal_mz_constant)

def find_closest_forbidden_zone(value, phospho_enriched):
        lower_forbidden = calculate_forbidden_zones(value - 1, phospho_enriched)
        upper_forbidden = calculate_forbidden_zones(value, phospho_enriched)

        return lower_forbidden if abs(value - lower_forbidden) < abs(value - upper_forbidden) else upper_forbidden

def adjust_bin_boundaries(bins, phospho_enriched=False):
    """Adjust bin boundaries to be closer to forbidden zones.

    Parameters:
    -----------
    bins : list of lists
        List of [start, end] bin boundaries
    phospho_enriched : bool, optional
        Whether the sample is phospho-enriched (default: False)

    Returns:
    --------
    list of lists
        Adjusted bin boundaries
    """

    adjusted_bins = []

    for i in range(len(bins)):
        start_value = bins[i][0]
        end_value = bins[i][1]

        adjusted_start = find_closest_forbidden_zone(start_value, phospho_enriched)
        adjusted_end = find_closest_forbidden_zone(end_value, phospho_enriched)

        adjusted_bins.append([adjusted_start, adjusted_end])

    return adjusted_bins


