#pragma once

/// @file distance.hpp
/// @brief Distance and proximity functions for multiobjective quality metrics.
///
/// Provides weighted Tchebycheff, Euclidean, p-norm, and epsilon-ratio
/// functions used by the quality indicators.
///
/// Part of the moqm (Multiobjective Quality Metrics) header-only library.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "point.hpp"

namespace moqm {

// ---------------------------------------------------------------------------
// Weight computation
// ---------------------------------------------------------------------------

/// @brief Compute range-based Tchebycheff weights: w_j = 1 / range_j.
///
/// For each objective j, the weight is the reciprocal of the range
/// (max_j - min_j) across the given point set.  If the range is below
/// @p eps, the weight defaults to 1.0 to avoid division by zero.
///
/// @param pts  The point set (typically Y_N).
/// @param eps  Minimum range threshold (default 1e-12).
/// @return     Weight vector of size m (number of objectives).
template <typename T>
[[nodiscard]] inline std::vector<double>
compute_range_weights(const PointSet<T> &pts, double eps = 1e-12) {
  if (pts.empty())
    return {};
  const std::size_t m = pts.front().dim();
  std::vector<double> lo(m, std::numeric_limits<double>::max());
  std::vector<double> hi(m, std::numeric_limits<double>::lowest());
  for (const auto &p : pts) {
    for (std::size_t j = 0; j < m; ++j) {
      lo[j] = std::min(lo[j], static_cast<double>(p[j]));
      hi[j] = std::max(hi[j], static_cast<double>(p[j]));
    }
  }
  std::vector<double> w(m);
  for (std::size_t j = 0; j < m; ++j) {
    double range = hi[j] - lo[j];
    w[j] = (range > eps) ? (1.0 / range) : 1.0;
  }
  return w;
}

/// @brief Helper to compute standard Tchebycheff weights from a point set.
///
/// Weight w_j = 1 / (max_j - min_j). If max_j == min_j, w_j = 1.0.
///
/// @param pts The point set to compute bounds from.
/// @return A vector of weights suitable for WeightedTchebycheffDistance.
template <typename T>
[[nodiscard]] inline std::vector<double>
compute_tchebycheff_weights(const PointSet<T> &pts) {
  if (pts.empty())
    return {};

  const std::size_t m = pts[0].dim();
  std::vector<double> mins(m, std::numeric_limits<double>::max());
  std::vector<double> maxs(m, std::numeric_limits<double>::lowest());

  for (const auto &p : pts) {
    for (std::size_t j = 0; j < m; ++j) {
      mins[j] = std::min(mins[j], static_cast<double>(p[j]));
      maxs[j] = std::max(maxs[j], static_cast<double>(p[j]));
    }
  }

  std::vector<double> weights(m, 1.0);
  for (std::size_t j = 0; j < m; ++j) {
    double range = maxs[j] - mins[j];
    if (range > 1e-12) {
      weights[j] = 1.0 / range;
    }
  }

  return weights;
}

// ---------------------------------------------------------------------------
// Range computation
// ---------------------------------------------------------------------------

/// @brief Compute per-objective min and max values of a point set.
///
/// @param pts  The point set.
/// @return     Pair of (min_vec, max_vec).
template <typename T>
[[nodiscard]] inline std::pair<std::vector<double>, std::vector<double>>
compute_ranges(const PointSet<T> &pts) {
  if (pts.empty())
    return {{}, {}};
  const std::size_t m = pts.front().dim();
  std::vector<double> lo(m, std::numeric_limits<double>::max());
  std::vector<double> hi(m, std::numeric_limits<double>::lowest());
  for (const auto &p : pts) {
    for (std::size_t j = 0; j < m; ++j) {
      lo[j] = std::min(lo[j], static_cast<double>(p[j]));
      hi[j] = std::max(hi[j], static_cast<double>(p[j]));
    }
  }
  return {lo, hi};
}

// ---------------------------------------------------------------------------
// Distance functions
// ---------------------------------------------------------------------------

/// @brief Weighted Tchebycheff distance (L∞ weighted).
///
///   d(a, b; w) = max_j  w_j * |a_j - b_j|
///
/// @param a  First point.
/// @param b  Second point.
/// @param w  Weight vector (must have size == a.dim()).
/// @return   The weighted Tchebycheff distance.
template <typename T>
[[nodiscard]] inline double weighted_tchebycheff(const Point<T> &a,
                                                 const Point<T> &b,
                                                 const std::vector<double> &w) {
  double d = 0.0;
  for (std::size_t j = 0; j < a.dim(); ++j) {
    d = std::max(d, w[j] * std::abs(static_cast<double>(a[j] - b[j])));
  }
  return d;
}

/// @brief Euclidean distance (L² norm).
///
///   d(a, b) = sqrt( sum_j (a_j - b_j)² )
template <typename T>
[[nodiscard]] inline double euclidean(const Point<T> &a, const Point<T> &b) {
  double s = 0.0;
  for (std::size_t j = 0; j < a.dim(); ++j) {
    double d = static_cast<double>(a[j]) - static_cast<double>(b[j]);
    s += d * d;
  }
  return std::sqrt(s);
}

/// @brief Lp-norm distance.
///
///   d(a, b) = ( sum_j |a_j - b_j|^p )^(1/p)
///
/// For p = infinity, use chebyshev() below.
template <typename T>
[[nodiscard]] inline double lp_norm(const Point<T> &a, const Point<T> &b,
                                    double p) {
  double s = 0.0;
  for (std::size_t j = 0; j < a.dim(); ++j) {
    s += std::pow(std::abs(static_cast<double>(a[j] - b[j])), p);
  }
  return std::pow(s, 1.0 / p);
}

/// @brief Chebyshev distance (L∞ norm, unweighted).
///
///   d(a, b) = max_j |a_j - b_j|
template <typename T>
[[nodiscard]] inline double chebyshev(const Point<T> &a, const Point<T> &b) {
  double d = 0.0;
  for (std::size_t j = 0; j < a.dim(); ++j) {
    d = std::max(d, std::abs(static_cast<double>(a[j] - b[j])));
  }
  return d;
}

// ---------------------------------------------------------------------------
// Epsilon-ratio (multiplicative approximation)
// ---------------------------------------------------------------------------

/// @brief Epsilon-ratio ε(r, b) parameterised by optimization sense.
///
/// For Maximize: ε(r, b) = max_i  b_i / r_i
/// For Minimize: ε(r, b) = max_i  r_i / b_i
///
/// Assumes all components are strictly positive.
///
/// @param r     Representative point.
/// @param b     Point to be covered.
/// @param sense optimization sense.
/// @return      The epsilon ratio.
template <typename T>
[[nodiscard]] inline double epsilon_ratio(const Point<T> &r, const Point<T> &b,
                                          Sense sense) {
  double eps = 0.0;
  for (std::size_t j = 0; j < r.dim(); ++j) {
    double rj = static_cast<double>(r[j]);
    double bj = static_cast<double>(b[j]);
    if (sense == Sense::Maximize) {
      if (rj <= 0.0)
        throw std::invalid_argument(
            "moqm::epsilon_ratio: representative component must be > 0.");
      eps = std::max(eps, bj / rj);
    } else {
      if (bj <= 0.0)
        throw std::invalid_argument(
            "moqm::epsilon_ratio: reference component must be > 0.");
      eps = std::max(eps, rj / bj);
    }
  }
  return eps;
}

// ---------------------------------------------------------------------------
// Callable distance objects (for use as template arguments)
// ---------------------------------------------------------------------------

/// @brief Functor: Euclidean distance.
struct EuclideanDistance {
  template <typename T>
  double operator()(const Point<T> &a, const Point<T> &b) const {
    return euclidean(a, b);
  }
};

/// @brief Functor: Chebyshev (L∞) distance.
struct ChebyshevDistance {
  template <typename T>
  double operator()(const Point<T> &a, const Point<T> &b) const {
    return chebyshev(a, b);
  }
};

/// @brief Functor: Weighted Tchebycheff distance.
///
/// Construct with a precomputed weight vector.
struct WeightedTchebycheffDistance {
  std::vector<double> weights;

  explicit WeightedTchebycheffDistance(std::vector<double> w)
      : weights(std::move(w)) {}

  template <typename T>
  explicit WeightedTchebycheffDistance(const PointSet<T> &reference_set)
      : weights(compute_tchebycheff_weights(reference_set)) {}

  template <typename T>
  double operator()(const Point<T> &a, const Point<T> &b) const {
    return weighted_tchebycheff(a, b, weights);
  }
};

/// @brief Functor: Lp-norm distance.
///
/// Construct with the desired p value.
struct LpNormDistance {
  double p;

  explicit LpNormDistance(double p_val) : p(p_val) {}

  template <typename T>
  double operator()(const Point<T> &a, const Point<T> &b) const {
    return lp_norm(a, b, p);
  }
};

/// @brief Functor: ε-ratio parameterised by sense.
struct EpsilonRatio {
  Sense sense;

  explicit EpsilonRatio(Sense s) : sense(s) {}

  template <typename T>
  double operator()(const Point<T> &r, const Point<T> &b) const {
    return epsilon_ratio(r, b, sense);
  }
};

} // namespace moqm
