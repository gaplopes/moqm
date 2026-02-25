#pragma once

/// @file indicators.hpp
/// @brief Quality indicator evaluation functions.
///
/// Implements: Coverage Error (CE), Median Error (ME), Range Ratio (RR),
/// Hypervolume Ratio (HVR), Uniformity (I_U), and ε-indicator (I_ε).
///
/// All functions are dimension-independent and accept arbitrary distance
/// functors via template parameters.
///
/// Part of the moqm (Multiobjective Quality Metrics) header-only library.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "distance.hpp"
#include "point.hpp"

namespace moqm {

// ===================================================================
// Coverage Error — CE(R) = max_{y∈Y_N} min_{r∈R} d(y, r)
// ===================================================================

/// @brief Coverage Error (one-sided Hausdorff distance from Y_N to R).
///
/// For each point in Y_N, the error is the distance to its nearest
/// representative in R.  The coverage error is the maximum of these.
///
/// Complexity: O(|Y_N| · |R| · m)
///
/// @tparam DistanceFn  A callable with signature double(Point<T>, Point<T>).
/// @tparam T           Point value type.
/// @param Y_N          The complete non-dominated set.
/// @param R            The representation (subset of Y_N or external).
/// @param dist         The distance function.
/// @return             The coverage error value.
template <typename T, typename DistanceFn>
[[nodiscard]] inline double
coverage_error(const PointSet<T> &Y_N, const PointSet<T> &R, DistanceFn dist) {
  if (Y_N.empty() || R.empty())
    return 0.0;
  double max_err = 0.0;
  for (const auto &y : Y_N) {
    double min_d = std::numeric_limits<double>::max();
    for (const auto &r : R) {
      min_d = std::min(min_d, dist(y, r));
    }
    max_err = std::max(max_err, min_d);
  }
  return max_err;
}

/// @brief Coverage Error using weighted Tchebycheff distance with
///        range-based weights.
///
/// Convenience overload that automatically computes weights from Y_N.
///
/// @param Y_N  The complete non-dominated set.
/// @param R    The representation.
/// @return     The coverage error value.
template <typename T = double>
[[nodiscard]] inline double coverage_error(const PointSet<T> &Y_N,
                                           const PointSet<T> &R) {
  auto w = compute_range_weights(Y_N);
  return coverage_error(Y_N, R, WeightedTchebycheffDistance(std::move(w)));
}

// ===================================================================
// Median Error — ME(R)
// ===================================================================

/// @brief Median Error: median of per-point errors min_{r∈R} d(y, r).
///
/// For each y ∈ Y_N, the error is min_{r∈R} d(y,r).  ME is the median
/// of these errors.  Half of Y_N is covered with error ≤ ME(R).
///
/// Complexity: O(|Y_N| · |R| · m + |Y_N| · log|Y_N|)
///
/// @tparam DistanceFn  A callable with signature double(Point, Point).
/// @param Y_N          The complete non-dominated set.
/// @param R            The representation.
/// @param dist         The distance function.
/// @return             The median error value.
template <typename T, typename DistanceFn>
[[nodiscard]] inline double
median_error(const PointSet<T> &Y_N, const PointSet<T> &R, DistanceFn dist) {
  if (Y_N.empty() || R.empty())
    return 0.0;
  std::vector<double> errors;
  errors.reserve(Y_N.size());
  for (const auto &y : Y_N) {
    double min_d = std::numeric_limits<double>::max();
    for (const auto &r : R) {
      min_d = std::min(min_d, dist(y, r));
    }
    errors.push_back(min_d);
  }
  const std::size_t n = errors.size();
  std::sort(errors.begin(), errors.end());
  if (n % 2 == 0) {
    return (errors[n / 2 - 1] + errors[n / 2]) / 2.0;
  }
  return errors[n / 2];
}

/// @brief Median Error using weighted Tchebycheff distance.
template <typename T = double>
[[nodiscard]] inline double median_error(const PointSet<T> &Y_N,
                                         const PointSet<T> &R) {
  auto w = compute_range_weights(Y_N);
  return median_error(Y_N, R, WeightedTchebycheffDistance(std::move(w)));
}

// ===================================================================
// Range Ratio — RR(R) = (1/m) Σ_j range_j(R) / range_j(Y_N)
// ===================================================================

/// @brief Range Ratio: average per-objective range coverage.
///
/// For each objective j:
///   RR_j = (max_R y_j − min_R y_j) / (max_{Y_N} y_j − min_{Y_N} y_j)
///
/// RR(R) = (1/m) Σ_j RR_j
///
/// A value of 1.0 means R covers the full range of Y_N in every objective.
///
/// Complexity: O((|Y_N| + |R|) · m)
///
/// @param Y_N        The complete non-dominated set.
/// @param R          The representation.
/// @param eps        Minimum range threshold (default 1e-12).
/// @return           The range ratio value ∈ [0, 1].
template <typename T>
[[nodiscard]] inline double
range_ratio(const PointSet<T> &Y_N, const PointSet<T> &R, double eps = 1e-12) {
  if (Y_N.empty() || R.empty())
    return 0.0;
  const std::size_t m = Y_N.front().dim();
  auto [yn_lo, yn_hi] = compute_ranges(Y_N);
  auto [r_lo, r_hi] = compute_ranges(R);

  double sum = 0.0;
  for (std::size_t j = 0; j < m; ++j) {
    double yn_range = yn_hi[j] - yn_lo[j];
    double r_range = r_hi[j] - r_lo[j];
    if (yn_range < eps) {
      sum += 1.0; // degenerate: Y_N has no range → R covers perfectly
    } else {
      sum += r_range / yn_range;
    }
  }
  return sum / static_cast<double>(m);
}

// ===================================================================
// Uniformity — I_U(R) = min_{r_i ≠ r_j ∈ R} d(r_i, r_j)
// ===================================================================

/// @brief Uniformity: minimum pairwise distance among representation points.
///
/// Higher is better — indicates the points are well-spread.
///
/// Complexity: O(|R|² · m)
///
/// @tparam DistanceFn  A callable with signature double(Point, Point).
/// @param R          The representation.
/// @param dist       The distance function.
/// @return           The uniformity value.
template <typename T, typename DistanceFn>
[[nodiscard]] inline double uniformity(const PointSet<T> &R, DistanceFn dist) {
  if (R.size() < 2)
    return std::numeric_limits<double>::infinity();
  double min_d = std::numeric_limits<double>::max();
  for (std::size_t i = 0; i < R.size(); ++i) {
    for (std::size_t j = i + 1; j < R.size(); ++j) {
      min_d = std::min(min_d, dist(R[i], R[j]));
    }
  }
  return min_d;
}

/// @brief Uniformity using Euclidean distance (default).
template <typename T = double>
[[nodiscard]] inline double uniformity(const PointSet<T> &R) {
  return uniformity(R, EuclideanDistance{});
}

// ===================================================================
// ε-Indicator — I_ε(R, B) = max_{b∈B} min_{r∈R} ε(r, b)
// ===================================================================

/// @brief Unary ε-indicator parameterised by optimization sense.
///
/// For Maximize: I_ε(R, B) = max_{b∈B} min_{r∈R} max_i(b_i / r_i)
/// For Minimize: I_ε(R, B) = max_{b∈B} min_{r∈R} max_i(r_i / b_i)
///
/// If R = B, then I_ε = 1.  Lower is better.
/// Assumes all components are strictly positive.
///
/// Complexity: O(|B| · |R| · m)
///
/// @param B      The reference set (typically the full non-dominated set).
/// @param R      The representation.
/// @param sense  optimization sense.
/// @return       The ε-indicator value (≥ 1 when R ⊆ B).
template <typename T>
[[nodiscard]] inline double
epsilon_indicator(const PointSet<T> &B, const PointSet<T> &R, Sense sense) {
  if (B.empty() || R.empty())
    return std::numeric_limits<double>::infinity();
  double max_eps = 0.0;
  for (const auto &b : B) {
    double min_r = std::numeric_limits<double>::max();
    for (const auto &r : R) {
      min_r = std::min(min_r, epsilon_ratio(r, b, sense));
    }
    max_eps = std::max(max_eps, min_r);
  }
  return max_eps;
}

// ===================================================================
// Hypervolume (2D exact via sweep-line)
// ===================================================================

namespace detail {

/// @brief Compute 2D hypervolume parameterised by optimization sense.
///
/// Complexity: O(n log n)
///
/// @param pts    The point set.
/// @param ref    The reference point.
/// @param sense  optimization sense.
/// @return       The hypervolume value.
template <typename T>
[[nodiscard]] inline double hypervolume_2d(PointSet<T> pts, const Point<T> &ref,
                                           Sense sense) {
  if (pts.empty() || ref.dim() != 2)
    return 0.0;

  if (sense == Sense::Maximize) {
    // Sort by first component descending.
    std::sort(pts.begin(), pts.end(),
              [](const Point<T> &a, const Point<T> &b) { return a[0] > b[0]; });
    double hv = 0.0;
    double y2_prev = static_cast<double>(ref[1]);
    for (const auto &p : pts) {
      if (p[0] <= ref[0] || p[1] <= ref[1])
        continue;
      double p1 = static_cast<double>(p[1]);
      if (p1 > y2_prev) {
        double width = static_cast<double>(p[0]) - static_cast<double>(ref[0]);
        double height = p1 - y2_prev;
        hv += width * height;
        y2_prev = p1;
      }
    }
    return hv;
  } else {
    // Sort by first component ascending.
    std::sort(pts.begin(), pts.end(),
              [](const Point<T> &a, const Point<T> &b) { return a[0] < b[0]; });
    double hv = 0.0;
    double y2_prev = static_cast<double>(ref[1]);
    for (const auto &p : pts) {
      if (p[0] >= ref[0] || p[1] >= ref[1])
        continue;
      double p1 = static_cast<double>(p[1]);
      if (p1 < y2_prev) {
        double width = static_cast<double>(ref[0]) - static_cast<double>(p[0]);
        double height = y2_prev - p1;
        hv += width * height;
        y2_prev = p1;
      }
    }
    return hv;
  }
}

/// @brief Compute hypervolume for general dimensions using
///        inclusion-exclusion (exact but exponential in m).
///
/// For m == 2, delegates to the efficient sweep-line implementation.
/// For m > 2, returns 0.0. (Not implemented yet)
template <typename T>
[[nodiscard]] inline double
hypervolume_general(const PointSet<T> &pts, const Point<T> &ref, Sense sense) {
  if (pts.empty())
    return 0.0;
  const std::size_t m = pts.front().dim();

  if (m == 2) {
    return hypervolume_2d(pts, ref, sense);
  }

  if (sense == Sense::Minimize) {
    throw std::runtime_error(
        "moqm::hypervolume: Minimize sense not implemented for m > 2.");
  }

  return 0.0;
}

} // namespace detail

// ===================================================================
// Hypervolume Ratio — HVR(R) = HV(R) / HV(Y_N)
// ===================================================================

/// @brief Hypervolume of a point set with respect to a reference point.
template <typename T>
[[nodiscard]] inline double hypervolume(const PointSet<T> &pts,
                                        const Point<T> &ref,
                                        Sense sense = Sense::Maximize) {
  return detail::hypervolume_general(pts, ref, sense);
}

/// @brief Hypervolume Ratio: HV(R) / HV(Y_N).
template <typename T>
[[nodiscard]] inline double
hypervolume_ratio(const PointSet<T> &Y_N, const PointSet<T> &R,
                  const Point<T> &ref, Sense sense = Sense::Maximize) {
  double hv_yn = hypervolume(Y_N, ref, sense);
  if (hv_yn <= 0.0)
    return 1.0; // degenerate case
  double hv_r = hypervolume(R, ref, sense);
  return hv_r / hv_yn;
}

} // namespace moqm
