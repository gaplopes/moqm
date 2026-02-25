#pragma once

/// @file point.hpp
/// @brief Dimension-agnostic point type for multiobjective quality metrics.
///
/// Part of the moqm (Multiobjective Quality Metrics) header-only library.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace moqm {

// ---------------------------------------------------------------------------
// Point<T>
// ---------------------------------------------------------------------------

/// @brief A point in m-dimensional objective space, parameterised on the
///        value type T (e.g. double, float, long double, int).
///
/// Wraps a std::vector<T> and provides dimension queries, lexicographic
/// ordering (exact, suitable for std::set / std::map), and approximate
/// equality for floating-point comparisons.
template <typename T = double> class Point {
public:
  using value_type = T;

  /// Construct from an initializer list.
  Point(std::initializer_list<T> il) : values_(il) {}

  /// Construct from a vector (copy).
  explicit Point(const std::vector<T> &v) : values_(v) {}

  /// Construct from a vector (move).
  explicit Point(std::vector<T> &&v) noexcept : values_(std::move(v)) {}

  /// Number of objectives (dimension).
  [[nodiscard]] std::size_t dim() const noexcept { return values_.size(); }

  /// Element access.
  [[nodiscard]] T operator[](std::size_t i) const { return values_[i]; }
  [[nodiscard]] T &operator[](std::size_t i) { return values_[i]; }

  /// Read-only access to the underlying vector.
  [[nodiscard]] const std::vector<T> &values() const noexcept {
    return values_;
  }

  /// Mutable access to the underlying vector.
  [[nodiscard]] std::vector<T> &values() noexcept { return values_; }

  // -- Exact lexicographic ordering (for use in std::set / std::map) --------

  friend bool operator<(const Point &a, const Point &b) noexcept {
    return a.values_ < b.values_; // std::vector lexicographic
  }
  friend bool operator==(const Point &a, const Point &b) noexcept {
    return a.values_ == b.values_;
  }
  friend bool operator!=(const Point &a, const Point &b) noexcept {
    return !(a == b);
  }
  friend bool operator>(const Point &a, const Point &b) noexcept {
    return b < a;
  }
  friend bool operator<=(const Point &a, const Point &b) noexcept {
    return !(b < a);
  }
  friend bool operator>=(const Point &a, const Point &b) noexcept {
    return !(a < b);
  }

  // -- Approximate equality -------------------------------------------------

  /// Check approximate equality within a given tolerance (epsilon).
  [[nodiscard]] bool approx_equal(const Point &other,
                                  T eps = T(1e-9)) const noexcept {
    if (values_.size() != other.values_.size())
      return false;
    for (std::size_t i = 0; i < values_.size(); ++i) {
      if (std::abs(values_[i] - other.values_[i]) > eps)
        return false;
    }
    return true;
  }

  // -- String representation ------------------------------------------------

  /// Human-readable string: "[v0, v1, ..., vm]".
  [[nodiscard]] std::string to_string(int precision = 6) const {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < values_.size(); ++i) {
      oss << std::fixed << std::setprecision(precision) << values_[i];
      if (i + 1 < values_.size())
        oss << ", ";
    }
    oss << "]";
    return oss.str();
  }

private:
  std::vector<T> values_;
};

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Default point type using double.
using PointD = Point<double>;

/// A set of points (ordered collection).
template <typename T = double> using PointSet = std::vector<Point<T>>;

/// Default point set type using double.
using PointSetD = PointSet<double>;

// ---------------------------------------------------------------------------
// Dominance utilities
// ---------------------------------------------------------------------------

/// @brief Sense of optimization.
enum class Sense { Minimize, Maximize };

/// @brief Check whether point @p a weakly dominates point @p b.
///
/// For Maximize: a weakly dominates b iff a_i >= b_i for all i.
/// For Minimize: a weakly dominates b iff a_i <= b_i for all i.
template <typename T>
[[nodiscard]] inline bool weakly_dominates(const Point<T> &a, const Point<T> &b,
                                           Sense sense) noexcept {
  if (a.dim() != b.dim())
    return false;
  for (std::size_t i = 0; i < a.dim(); ++i) {
    if (sense == Sense::Maximize) {
      if (a[i] < b[i])
        return false;
    } else {
      if (a[i] > b[i])
        return false;
    }
  }
  return true;
}

/// @brief Check whether point @p a strictly dominates point @p b.
///
/// a dominates b iff a weakly dominates b AND a != b.
template <typename T>
[[nodiscard]] inline bool dominates(const Point<T> &a, const Point<T> &b,
                                    Sense sense) noexcept {
  if (!weakly_dominates(a, b, sense))
    return false;
  for (std::size_t i = 0; i < a.dim(); ++i) {
    if (a[i] != b[i])
      return true; // at least one strictly better
  }
  return false; // they are equal
}

// ---------------------------------------------------------------------------
// Sorting utilities
// ---------------------------------------------------------------------------

/// @brief Sort a PointSet by the first component (ascending).
///
/// This is a precondition for all biobjective DP algorithms.
template <typename T> inline void sort_by_first_component(PointSet<T> &pts) {
  std::sort(pts.begin(), pts.end(),
            [](const Point<T> &a, const Point<T> &b) { return a[0] < b[0]; });
}

} // namespace moqm
