#pragma once

/*
 * Histogram Implementation
 *
 * MATHEMATICAL DEFINITION:
 * ========================
 * A histogram counts the number of data points falling into discrete bins.
 *
 * For a dataset x_1, x_2, ..., x_n and bin edges e_0, e_1, ..., e_k:
 *   - Bin i contains values in range (e_{i-1}, e_i] for closed="right"
 *   - Bin i contains values in range [e_{i-1}, e_i) for closed="left"
 *
 * ALGORITHM:
 * ==========
 * 1. Compute bin edges from range and number of bins (equal-width binning)
 * 2. For each data point, use binary search to find the appropriate bin
 * 3. Scatter-add counts (or weights if weighted)
 * 4. Optionally normalize to density: counts / (total_count * bin_width)
 *
 * EDGE CASES:
 * ===========
 * - NaN values are ignored (not counted in any bin)
 * - Out-of-bounds handling:
 *   - "clamp": values outside range are placed in first/last bins
 *   - "ignore": values outside range are not counted
 */

#include <c10/macros/Macros.h>
#include <cmath>
#include <limits>

namespace torchscience::impl::descriptive {

// ============================================================================
// Constants
// ============================================================================

// Small epsilon for floating point comparisons
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T histogram_epsilon() { return T(1e-10); }

// ============================================================================
// Bin Edge Computation
// ============================================================================

/**
 * Compute equal-width bin edges from data range.
 *
 * @param edges Output array of (num_bins + 1) elements for bin edges
 * @param data_min Minimum value of data range
 * @param data_max Maximum value of data range
 * @param num_bins Number of bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void compute_bin_edges(
    T* edges,
    T data_min,
    T data_max,
    int64_t num_bins
) {
    T bin_width = (data_max - data_min) / static_cast<T>(num_bins);

    for (int64_t i = 0; i <= num_bins; ++i) {
        edges[i] = data_min + static_cast<T>(i) * bin_width;
    }

    // Ensure the last edge is exactly data_max to avoid floating point issues
    edges[num_bins] = data_max;
}

/**
 * Compute bin width from edges.
 *
 * @param data_min Minimum value of data range
 * @param data_max Maximum value of data range
 * @param num_bins Number of bins
 * @return Width of each bin
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T compute_bin_width(T data_min, T data_max, int64_t num_bins) {
    return (data_max - data_min) / static_cast<T>(num_bins);
}

// ============================================================================
// Bin Finding
// ============================================================================

/**
 * Find bin index for a value using binary search on pre-computed edges.
 *
 * For closed="right" (default): bin i contains (edge[i], edge[i+1]]
 * For closed="left": bin i contains [edge[i], edge[i+1])
 *
 * @param value The value to place in a bin
 * @param edges Array of (num_bins + 1) bin edges
 * @param num_bins Number of bins
 * @param closed_right True for (a, b] intervals, false for [a, b) intervals
 * @return Bin index (0 to num_bins-1), or -1 if value is below range,
 *         or num_bins if value is above range
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
int64_t find_bin(
    T value,
    const T* edges,
    int64_t num_bins,
    bool closed_right
) {
    // Handle NaN - return special value
    if (value != value) {  // NaN check
        return -2;  // Special marker for NaN
    }

    T left_edge = edges[0];
    T right_edge = edges[num_bins];

    // Check out of bounds
    if (closed_right) {
        // (a, b] - value at left edge goes to first bin only if it equals left_edge exactly
        if (value <= left_edge) {
            // For closed right, left boundary is special case
            if (value == left_edge) {
                return 0;  // Include left boundary in first bin
            }
            return -1;  // Below range
        }
        if (value > right_edge) {
            return num_bins;  // Above range
        }
    } else {
        // [a, b)
        if (value < left_edge) {
            return -1;  // Below range
        }
        if (value >= right_edge) {
            // For closed left, right boundary is special case
            if (value == right_edge) {
                return num_bins - 1;  // Include right boundary in last bin
            }
            return num_bins;  // Above range
        }
    }

    // Binary search to find the bin
    int64_t lo = 0;
    int64_t hi = num_bins;

    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;

        if (closed_right) {
            // Looking for bin where edge[mid] < value <= edge[mid+1]
            if (edges[mid + 1] < value) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        } else {
            // Looking for bin where edge[mid] <= value < edge[mid+1]
            if (edges[mid + 1] <= value) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
    }

    return lo;
}

/**
 * Find bin index for a value given min, max, and num_bins (equal-width bins).
 * More efficient than pre-computing edges for simple equal-width case.
 *
 * @param value The value to place in a bin
 * @param data_min Minimum value of data range
 * @param data_max Maximum value of data range
 * @param num_bins Number of bins
 * @param closed_right True for (a, b] intervals, false for [a, b) intervals
 * @return Bin index (0 to num_bins-1), or -1 if value is below range,
 *         or num_bins if value is above range
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
int64_t find_bin_equal_width(
    T value,
    T data_min,
    T data_max,
    int64_t num_bins,
    bool closed_right
) {
    // Handle NaN
    if (value != value) {
        return -2;
    }

    // Check out of bounds
    if (closed_right) {
        if (value <= data_min) {
            if (value == data_min) {
                return 0;  // Include left boundary in first bin
            }
            return -1;
        }
        if (value > data_max) {
            return num_bins;
        }
    } else {
        if (value < data_min) {
            return -1;
        }
        if (value >= data_max) {
            if (value == data_max) {
                return num_bins - 1;  // Include right boundary in last bin
            }
            return num_bins;
        }
    }

    // Direct computation for equal-width bins
    T bin_width = (data_max - data_min) / static_cast<T>(num_bins);
    int64_t bin_idx = static_cast<int64_t>((value - data_min) / bin_width);

    // Handle edge case where value == data_max exactly for closed_right
    if (bin_idx >= num_bins) {
        bin_idx = num_bins - 1;
    }

    // For closed_right, adjust if value is exactly on the left edge of a bin
    if (closed_right && bin_idx > 0) {
        T edge = data_min + static_cast<T>(bin_idx) * bin_width;
        if (value <= edge) {
            bin_idx -= 1;
        }
    }

    return bin_idx;
}

// ============================================================================
// Main 1D Histogram Computation
// ============================================================================

/**
 * Compute 1D histogram for a contiguous array.
 *
 * @param data Input array of n elements
 * @param n Number of data elements
 * @param counts Output array of num_bins elements (histogram counts)
 * @param edges Bin edges array of (num_bins + 1) elements
 * @param num_bins Number of bins
 * @param weights Optional weight array (nullptr for unweighted)
 * @param closed_right True for (a, b] intervals, false for [a, b) intervals
 * @param clamp_out_of_bounds If true, clamp out-of-bounds values to edge bins
 * @return Total count (or sum of weights) of values placed in bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T histogram_1d(
    const T* data,
    int64_t n,
    T* counts,
    const T* edges,
    int64_t num_bins,
    const T* weights,
    bool closed_right,
    bool clamp_out_of_bounds
) {
    // Initialize counts to zero
    for (int64_t i = 0; i < num_bins; ++i) {
        counts[i] = T(0);
    }

    T total = T(0);

    for (int64_t i = 0; i < n; ++i) {
        T value = data[i];
        T weight = weights ? weights[i] : T(1);

        int64_t bin_idx = find_bin(value, edges, num_bins, closed_right);

        // Handle special cases
        if (bin_idx == -2) {
            // NaN value - ignore
            continue;
        }

        if (bin_idx == -1) {
            // Below range
            if (clamp_out_of_bounds) {
                bin_idx = 0;
            } else {
                continue;  // Ignore
            }
        } else if (bin_idx == num_bins) {
            // Above range
            if (clamp_out_of_bounds) {
                bin_idx = num_bins - 1;
            } else {
                continue;  // Ignore
            }
        }

        counts[bin_idx] += weight;
        total += weight;
    }

    return total;
}

/**
 * Compute 1D histogram using equal-width bins (more efficient).
 *
 * @param data Input array of n elements
 * @param n Number of data elements
 * @param counts Output array of num_bins elements (histogram counts)
 * @param data_min Minimum value of data range
 * @param data_max Maximum value of data range
 * @param num_bins Number of bins
 * @param weights Optional weight array (nullptr for unweighted)
 * @param closed_right True for (a, b] intervals, false for [a, b) intervals
 * @param clamp_out_of_bounds If true, clamp out-of-bounds values to edge bins
 * @return Total count (or sum of weights) of values placed in bins
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T histogram_1d_equal_width(
    const T* data,
    int64_t n,
    T* counts,
    T data_min,
    T data_max,
    int64_t num_bins,
    const T* weights,
    bool closed_right,
    bool clamp_out_of_bounds
) {
    // Initialize counts to zero
    for (int64_t i = 0; i < num_bins; ++i) {
        counts[i] = T(0);
    }

    T total = T(0);

    for (int64_t i = 0; i < n; ++i) {
        T value = data[i];
        T weight = weights ? weights[i] : T(1);

        int64_t bin_idx = find_bin_equal_width(value, data_min, data_max, num_bins, closed_right);

        // Handle special cases
        if (bin_idx == -2) {
            // NaN value - ignore
            continue;
        }

        if (bin_idx == -1) {
            // Below range
            if (clamp_out_of_bounds) {
                bin_idx = 0;
            } else {
                continue;  // Ignore
            }
        } else if (bin_idx == num_bins) {
            // Above range
            if (clamp_out_of_bounds) {
                bin_idx = num_bins - 1;
            } else {
                continue;  // Ignore
            }
        }

        counts[bin_idx] += weight;
        total += weight;
    }

    return total;
}

// ============================================================================
// Density Normalization
// ============================================================================

/**
 * Apply density normalization to histogram counts.
 *
 * Density normalization converts counts to probability density:
 *   density[i] = count[i] / (total_count * bin_width[i])
 *
 * For equal-width bins, all bin_widths are the same.
 * The resulting histogram integrates to 1.0.
 *
 * @param counts Array of histogram counts (modified in place)
 * @param num_bins Number of bins
 * @param total_count Total count (or sum of weights)
 * @param bin_width Width of bins (for equal-width histograms)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void normalize_density(
    T* counts,
    int64_t num_bins,
    T total_count,
    T bin_width
) {
    // Handle edge cases
    if (total_count <= T(0) || bin_width <= T(0)) {
        // Set all densities to zero if invalid
        for (int64_t i = 0; i < num_bins; ++i) {
            counts[i] = T(0);
        }
        return;
    }

    T normalizer = total_count * bin_width;

    for (int64_t i = 0; i < num_bins; ++i) {
        counts[i] /= normalizer;
    }
}

/**
 * Apply density normalization to histogram counts with variable bin widths.
 *
 * @param counts Array of histogram counts (modified in place)
 * @param edges Array of (num_bins + 1) bin edges
 * @param num_bins Number of bins
 * @param total_count Total count (or sum of weights)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void normalize_density_variable_width(
    T* counts,
    const T* edges,
    int64_t num_bins,
    T total_count
) {
    // Handle edge case
    if (total_count <= T(0)) {
        for (int64_t i = 0; i < num_bins; ++i) {
            counts[i] = T(0);
        }
        return;
    }

    for (int64_t i = 0; i < num_bins; ++i) {
        T bin_width = edges[i + 1] - edges[i];
        if (bin_width > T(0)) {
            counts[i] /= (total_count * bin_width);
        } else {
            counts[i] = T(0);
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Compute data range (min and max) for a 1D array, ignoring NaN values.
 *
 * @param data Input array
 * @param n Number of elements
 * @param out_min Output minimum value
 * @param out_max Output maximum value
 * @return True if valid range found, false if all values are NaN
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
bool compute_data_range(
    const T* data,
    int64_t n,
    T* out_min,
    T* out_max
) {
    T min_val = std::numeric_limits<T>::infinity();
    T max_val = -std::numeric_limits<T>::infinity();
    bool found_valid = false;

    for (int64_t i = 0; i < n; ++i) {
        T val = data[i];
        // Skip NaN values
        if (val != val) {
            continue;
        }
        found_valid = true;
        if (val < min_val) {
            min_val = val;
        }
        if (val > max_val) {
            max_val = val;
        }
    }

    if (!found_valid) {
        *out_min = std::numeric_limits<T>::quiet_NaN();
        *out_max = std::numeric_limits<T>::quiet_NaN();
        return false;
    }

    *out_min = min_val;
    *out_max = max_val;
    return true;
}

/**
 * Parse out_of_bounds string to boolean.
 *
 * @param out_of_bounds String: "clamp" or "ignore"
 * @return True for clamp, false for ignore
 */
C10_HOST_DEVICE C10_ALWAYS_INLINE
bool parse_out_of_bounds(const char* out_of_bounds) {
    // Simple comparison: "clamp" starts with 'c', "ignore" starts with 'i'
    return out_of_bounds[0] == 'c' || out_of_bounds[0] == 'C';
}

/**
 * Parse closed string to boolean.
 *
 * @param closed String: "left" or "right"
 * @return True for closed_right (a, b], false for closed_left [a, b)
 */
C10_HOST_DEVICE C10_ALWAYS_INLINE
bool parse_closed(const char* closed) {
    // "right" starts with 'r', "left" starts with 'l'
    return closed[0] == 'r' || closed[0] == 'R';
}

}  // namespace torchscience::impl::descriptive
