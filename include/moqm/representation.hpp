#pragma once

/// @file representation.hpp
/// @brief O(kn) dynamic programming algorithms for finding optimal
///        k-element representations of biobjective non-dominated sets.
///
/// Implements three algorithms from:
///   Vaz, Paquete, Fonseca, Klamroth, Stiglmayr.
///   "Representation of the non-dominated set in biobjective discrete
///    optimization." Computers & Operations Research 63 (2015): 172–186.
///
/// Algorithms:
///   1. dp_max_uniformity   — maximize min pairwise distance    (§3.1)
///   2. dp_min_coverage     — minimize max coverage radius      (§4.1)
///   3. dp_min_epsilon      — minimize ε-indicator              (§5.1)
///   4. thresh_max_uniformity — maximize min pairwise distance  (§3.2)
///   5. thresh_min_coverage — minimize max coverage radius      (§4.2)
///   6. thresh_min_epsilon — minimize ε-indicator               (§5.2)
///
/// Stub functions (not yet implemented):
///   7. dp_coverage_uniformity           — §6.1
///   8. dp_epsilon_uniformity            — §6.2
///   9. dp_coverage_epsilon              — §6.3
///   10. dp_coverage_epsilon_uniformity  — §6.4
///
/// All three exploit Proposition 2.1 (distance monotonicity for sorted
/// biobjective non-dominated points) to achieve O(kn) time complexity.
///
/// IMPORTANT: These algorithms are valid ONLY for biobjective (m=2) problems.
/// For m ≥ 3, the underlying k-center and k-dispersion problems are NP-hard.
///
/// Part of the moqm (Multiobjective Quality Metrics) header-only library.
/// This header has NO external dependencies.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "distance.hpp"
#include "point.hpp"

namespace moqm {

/// @brief Result of a representation problem solver.
template <typename T = double> struct RepresentationResult {
  double value;       ///< Optimal indicator value.
  PointSet<T> subset; ///< The selected k-element representation.
};

namespace detail {

// -----------------------------------------------------------------------
// Helper: ensure biobjective and sorted
// -----------------------------------------------------------------------

/// @brief Prepare a point set for biobjective DP: verify m == 2 and sort.
template <typename T> inline PointSet<T> prepare_biobjective(PointSet<T> B) {
  if (B.empty()) {
    throw std::invalid_argument("moqm: point set B must be non-empty.");
  }
  if (B.front().dim() != 2) {
    throw std::invalid_argument(
        "moqm: biobjective DP solvers require m = 2 objectives. "
        "Got m = " +
        std::to_string(B.front().dim()) + ".");
  }
  sort_by_first_component(B);
  return B;
}

/// @brief Validate k parameter.
inline void validate_k(std::size_t k, std::size_t n) {
  if (k == 0 || k > n) {
    throw std::invalid_argument(
        "moqm: k must satisfy 1 <= k <= |B|. Got k=" + std::to_string(k) +
        ", |B|=" + std::to_string(n) + ".");
  }
}

// -----------------------------------------------------------------------
// Backtrack helper: reconstruct subset from parent pointers
// -----------------------------------------------------------------------

/// @brief Backtrack solution from a parent matrix.
///
/// @param parent  parent[i][j] = ℓ chosen when computing T[i][j].
/// @param B       The sorted point set.
/// @param k       Cardinality of representation.
/// @param start_j The starting column index for row k.
/// @return        The selected k-element subset.
template <typename T>
inline PointSet<T>
backtrack(const std::vector<std::vector<std::size_t>> &parent,
          const PointSet<T> &B, std::size_t k, std::size_t start_j) {
  PointSet<T> result;
  result.reserve(k);
  std::size_t j = start_j;
  for (std::size_t i = k; i >= 1; --i) {
    result.push_back(B[j]);
    if (i > 1) {
      j = parent[i][j];
    }
  }
  std::reverse(result.begin(), result.end());
  return result;
}

} // namespace detail

// ===================================================================
// 1. Uniformity DP — maximize I_U(R)  (§3.1, Wang & Kuo)
// ===================================================================

/// @brief Find the k-element subset of B that maximizes uniformity.
///
/// Uniformity = min pairwise distance.  Goal: maximize this.
///
/// Algorithm (Wang & Kuo, adapted by Vaz et al.):
///   T(1,j) = +∞
///   T(i,j) = max_{j<ℓ≤n-i+2} min( ‖b_j − b_ℓ‖, T(i-1,ℓ) )
///   Result = max_{1≤j≤n-k+1} T(k,j)
///
/// ℓ-improvement: since T(i-1,ℓ) is non-increasing and ‖b_j−b_ℓ‖
/// is increasing, stop when min(...) starts decreasing → O(kn).
///
/// Complexity: O(kn + n log n)
///
/// @tparam T           Point value type.
/// @tparam DistanceFn  Callable with signature double(Point<T>, Point<T>).
/// @param B            Non-dominated set (must be biobjective, m=2).
/// @param k            Desired cardinality of representation.
/// @param dist         Distance function (default: Euclidean).
/// @return             RepresentationResult with optimal value and subset.
template <typename T = double, typename DistanceFn = EuclideanDistance>
[[nodiscard]] inline RepresentationResult<T>
dp_max_uniformity(PointSet<T> B, std::size_t k,
                  DistanceFn dist = DistanceFn{}) {
  B = detail::prepare_biobjective(std::move(B));
  const std::size_t n = B.size();
  detail::validate_k(k, n);

  // Special case: k == 1 → uniformity is +∞ by convention.
  if (k == 1) {
    return {std::numeric_limits<double>::infinity(), {B.front()}};
  }

  constexpr double INF = std::numeric_limits<double>::infinity();

  // Full DP matrix for backtracking.
  std::vector<std::vector<double>> DP(k + 1, std::vector<double>(n, 0.0));
  std::vector<std::vector<std::size_t>> parent(k + 1,
                                               std::vector<std::size_t>(n, 0));

  // Base case: DP(1,j) = +∞
  for (std::size_t j = 0; j < n; ++j)
    DP[1][j] = INF;

  // Fill rows i = 2, ..., k
  for (std::size_t i = 2; i <= k; ++i) {
    for (std::size_t j = 0; j <= n - i; ++j) {
      double best = 0.0;
      std::size_t best_ell = j + 1;

      for (std::size_t ell = j + 1; ell <= n - i + 1; ++ell) {
        double d_j_ell = dist(B[j], B[ell]);
        double val = std::min(d_j_ell, DP[i - 1][ell]);

        if (val > best) {
          best = val;
          best_ell = ell;
        }

        // ℓ-improvement
        if (d_j_ell >= DP[i - 1][ell])
          break;
      }

      DP[i][j] = best;
      parent[i][j] = best_ell;
    }
  }

  // Extract result: max_{0≤j≤n-k} DP[k][j]
  double opt_val = 0.0;
  std::size_t opt_j = 0;
  for (std::size_t j = 0; j <= n - k; ++j) {
    if (DP[k][j] > opt_val) {
      opt_val = DP[k][j];
      opt_j = j;
    }
  }

  auto subset = detail::backtrack(parent, B, k, opt_j);
  return {opt_val, std::move(subset)};
}

// ===================================================================
// 2. Coverage DP — minimize I_C(R, B)  (§4.1 with §4.1.3 improvement)
// ===================================================================

/// @brief Find the k-element subset of B that minimizes coverage.
///
/// Coverage = max_{b∈B} min_{r∈R} ‖r − b‖.  Goal: minimize this.
///
/// Algorithm (Vaz et al., §4.1, with m/ℓ-improvement §4.1.3):
///   T(1,j) = ‖b_j − b_n‖
///   δ_{j,ℓ} = max_{j≤m≤ℓ} min(‖b_j−b_m‖, ‖b_m−b_ℓ‖)
///   T(i,j) = min_{j<ℓ≤n-i+2} max(δ_{j,ℓ}, T(i-1,ℓ))
///   Correction: T̄(k,j) = max(‖b_1−b_j‖, T(k,j))
///   Result = min_{1≤j≤n-k+1} T̄(k,j)
///
/// The §4.1.3 improvement tracks m globally to avoid preprocessing,
/// achieving O(kn) total while computing δ on-the-fly.
///
/// Complexity: O(kn + n log n)
///
/// @tparam DistanceFn  Callable with signature double(Point, Point).
/// @param B            Non-dominated set (must be biobjective, m=2).
/// @param k            Desired cardinality of representation.
/// @param dist         Distance function (default: Euclidean).
/// @return             RepresentationResult with optimal value and subset.
template <typename T = double, typename DistanceFn = EuclideanDistance>
[[nodiscard]] inline RepresentationResult<T>
dp_min_coverage(PointSet<T> B, std::size_t k, DistanceFn dist = DistanceFn{}) {
  B = detail::prepare_biobjective(std::move(B));
  const std::size_t n = B.size();
  detail::validate_k(k, n);

  // Special case: k >= n → coverage is 0 (every point represents itself).
  if (k >= n) {
    return {0.0, B};
  }

  // Special case: k == 1.
  if (k == 1) {
    // Best single representative: the one minimizing max distance to b_1 and
    // b_n. We evaluate all candidates.
    double best_val = std::numeric_limits<double>::max();
    std::size_t best_j = 0;
    for (std::size_t j = 0; j < n; ++j) {
      // Coverage when selecting only b_j: max over all b of d(b_j, b).
      // Since sorted, max is max(d(b_j, b_0), d(b_j, b_{n-1})).
      double val = std::max(dist(B[j], B[0]), dist(B[j], B[n - 1]));
      if (val < best_val) {
        best_val = val;
        best_j = j;
      }
    }
    return {best_val, {B[best_j]}};
  }

  constexpr double INF = std::numeric_limits<double>::max();

  std::vector<std::vector<double>> DP(k + 1, std::vector<double>(n, INF));
  std::vector<std::vector<std::size_t>> parent(k + 1,
                                               std::vector<std::size_t>(n, 0));

  // Base case: DP(1, j) = ‖b_j − b_{n-1}‖
  for (std::size_t j = 0; j < n; ++j) {
    DP[1][j] = dist(B[j], B[n - 1]);
  }

  // Fill rows i = 2, ..., k
  for (std::size_t i = 2; i <= k; ++i) {
    // Global m tracking (§4.1.3): m* is non-decreasing in both j and ℓ.
    // Since we process j from right to left and ℓ from left to right,
    // we reset m_star when j changes.
    for (std::size_t j = 0; j <= n - i; ++j) {
      double best = INF;
      std::size_t best_ell = j + 1;
      std::size_t m_star = j; // starting m for δ computation

      for (std::size_t ell = j + 1; ell <= n - i + 1; ++ell) {
        // Compute δ_{j, ell} using m-improvement (§4.1.1/§4.1.3):
        // m_star only increases as ell increases (monotonicity).
        // Ensure m_star is in [j, ell].
        if (m_star < j)
          m_star = j;
        if (m_star > ell)
          m_star = ell;

        double delta = 0.0;
        // Start from current m_star and scan forward
        for (std::size_t m = m_star; m <= ell; ++m) {
          double d_jm = dist(B[j], B[m]);
          double d_mell = dist(B[m], B[ell]);
          double val = std::min(d_jm, d_mell);
          if (val > delta) {
            delta = val;
            m_star = m;
          }
          // If d_jm >= d_mell, further m increases will only make d_jm
          // larger and d_mell smaller, so min decreases → stop.
          if (d_jm >= d_mell)
            break;
        }

        double coverage = std::max(delta, DP[i - 1][ell]);

        if (coverage < best) {
          best = coverage;
          best_ell = ell;
        }

        // ℓ-improvement (§4.1.2): δ_{j,ℓ} is non-decreasing in ℓ,
        // T(i-1,ℓ) is non-increasing. When their max starts increasing
        // (i.e., δ > T), further ℓ only makes it worse → stop.
        if (delta >= DP[i - 1][ell])
          break;
      }

      DP[i][j] = best;
      parent[i][j] = best_ell;
    }
  }

  // Correction: DP̄(k,j) = max(‖b_0 − b_j‖, DP(k,j))
  double opt_val = INF;
  std::size_t opt_j = 0;
  for (std::size_t j = 0; j <= n - k; ++j) {
    double corrected = std::max(dist(B[0], B[j]), DP[k][j]);
    if (corrected < opt_val) {
      opt_val = corrected;
      opt_j = j;
    }
  }

  auto subset = detail::backtrack(parent, B, k, opt_j);
  return {opt_val, std::move(subset)};
}

// ===================================================================
// 3. ε-Indicator DP — minimize I_ε(R, B)  (§5.1)
// ===================================================================

/// @brief Find the k-element subset of B that minimizes the ε-indicator.
///
/// ε-indicator = max_{b∈B} min_{r∈R} max_i(b_i / r_i).  Goal: minimize.
///
/// Same structure as coverage DP but uses ε-ratio instead of norms.
///
///   T(1,j) = ε(b_j, b_{n-1})
///   δ_{j,ℓ} = max_{j≤m≤ℓ} min(ε(b_j, b_m), ε(b_ℓ, b_m))
///   T(i,j) = min_{j<ℓ≤n-i+2} max(δ_{j,ℓ}, T(i-1,ℓ))
///   Correction: T̄(k,j) = max(ε(b_j, b_0), T(k,j))
///   Result = min_{j} T̄(k,j)
///
/// The same m/ℓ-improvements apply by Propositions 2.3 and 2.4.
///
/// Complexity: O(kn + n log n)
///
/// @param B     Non-dominated set (must be biobjective, m=2, all positive).
/// @param k     Desired cardinality of representation.
/// @param sense optimization sense (Maximize or Minimize).
/// @return      RepresentationResult with optimal ε-indicator and subset.
template <typename T = double>
[[nodiscard]] inline RepresentationResult<T>
dp_min_epsilon(PointSet<T> B, std::size_t k, Sense sense = Sense::Maximize) {
  B = detail::prepare_biobjective(std::move(B));
  const std::size_t n = B.size();
  detail::validate_k(k, n);

  auto eps = [sense](const Point<T> &r, const Point<T> &b) -> double {
    return epsilon_ratio(r, b, sense);
  };

  // Special case: k >= n → ε-indicator is 1.0 (R = B).
  if (k >= n) {
    return {1.0, B};
  }

  // Special case: k == 1.
  if (k == 1) {
    double best_val = std::numeric_limits<double>::max();
    std::size_t best_j = 0;
    for (std::size_t j = 0; j < n; ++j) {
      // ε-indicator when selecting only b_j:
      // max over all b of ε(b_j, b).
      double val = 0.0;
      for (std::size_t m = 0; m < n; ++m) {
        val = std::max(val, eps(B[j], B[m]));
      }
      if (val < best_val) {
        best_val = val;
        best_j = j;
      }
    }
    return {best_val, {B[best_j]}};
  }

  constexpr double INF = std::numeric_limits<double>::max();

  std::vector<std::vector<double>> DP(k + 1, std::vector<double>(n, INF));
  std::vector<std::vector<std::size_t>> parent(k + 1,
                                               std::vector<std::size_t>(n, 0));

  // Base case: DP(1, j) = ε(b_j, b_{n-1})
  for (std::size_t j = 0; j < n; ++j) {
    DP[1][j] = eps(B[j], B[n - 1]);
  }

  // Fill rows i = 2, ..., k
  for (std::size_t i = 2; i <= k; ++i) {
    for (std::size_t j = 0; j <= n - i; ++j) {
      double best = INF;
      std::size_t best_ell = j + 1;
      std::size_t m_star = j;

      for (std::size_t ell = j + 1; ell <= n - i + 1; ++ell) {
        if (m_star < j)
          m_star = j;
        if (m_star > ell)
          m_star = ell;

        // Compute δ_{j, ell} for ε-indicator:
        // δ = max_{j≤m≤ell} min(ε(b_j, b_m), ε(b_ell, b_m))
        double delta = 0.0;
        for (std::size_t m = m_star; m <= ell; ++m) {
          double e_jm = eps(B[j], B[m]);
          double e_ellm = eps(B[ell], B[m]);
          double val = std::min(e_jm, e_ellm);
          if (val > delta) {
            delta = val;
            m_star = m;
          }
          // Analogous monotonicity: ε(b_j, b_m) increasing, ε(b_ell, b_m)
          // decreasing as m increases → stop when ε(b_j,b_m) ≥ ε(b_ell,b_m)
          if (e_jm >= e_ellm)
            break;
        }

        double indicator = std::max(delta, DP[i - 1][ell]);

        if (indicator < best) {
          best = indicator;
          best_ell = ell;
        }

        if (delta >= DP[i - 1][ell])
          break;
      }

      DP[i][j] = best;
      parent[i][j] = best_ell;
    }
  }

  // Correction: DP̄(k,j) = max(ε(b_j, b_0), DP(k,j))
  double opt_val = INF;
  std::size_t opt_j = 0;
  for (std::size_t j = 0; j <= n - k; ++j) {
    double corrected = std::max(eps(B[j], B[0]), DP[k][j]);
    if (corrected < opt_val) {
      opt_val = corrected;
      opt_j = j;
    }
  }

  auto subset = detail::backtrack(parent, B, k, opt_j);
  return {opt_val, std::move(subset)};
}

// ===================================================================
// 4–6. Threshold + greedy solvers (exact for m=2)
// ===================================================================
//
// Binary search on candidate threshold values + greedy selection.
// The greedy is exact for sorted biobjective non-dominated points
// due to Proposition 2.1 (distance monotonicity).
//
// Complexity O(n² log n) — slower than O(kn) DP, but useful for
// cross-validation of the DP results.

/// @brief Find the k-element subset that maximizes uniformity via
///        binary search on threshold distances + greedy selection.
///
/// Exact for m=2 (sorted biobjective nondominated points).
/// Complexity: O(n² log n)
///
/// @param B     Non-dominated set (must be biobjective, m=2).
/// @param k     Desired cardinality.
/// @param dist  Distance function (default: Euclidean).
/// @return      RepresentationResult with optimal uniformity and subset.
template <typename T = double, typename DistanceFn = EuclideanDistance>
[[nodiscard]] inline RepresentationResult<T>
thresh_max_uniformity(PointSet<T> B, std::size_t k,
                      DistanceFn dist = DistanceFn{}) {
  B = detail::prepare_biobjective(std::move(B));
  const std::size_t n = B.size();
  detail::validate_k(k, n);

  if (k == 1) {
    return {std::numeric_limits<double>::infinity(), {B.front()}};
  }

  // Precompute pairwise distances
  std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 1; j < n; ++j) {
      D[i][j] = D[j][i] = dist(B[i], B[j]);
    }

  // Collect unique pairwise distances as candidate thresholds
  std::vector<double> thresholds;
  thresholds.reserve(n * (n - 1) / 2);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 1; j < n; ++j)
      thresholds.push_back(D[i][j]);
  std::sort(thresholds.begin(), thresholds.end());
  thresholds.erase(
      std::unique(thresholds.begin(), thresholds.end(),
                  [](double a, double b) { return std::abs(a - b) < 1e-12; }),
      thresholds.end());

  // Greedy feasibility: can we pick k points with pairwise distances ≥ t?
  auto is_feasible = [&](double t) -> bool {
    std::vector<std::size_t> sel;
    sel.reserve(k);
    sel.push_back(0);
    for (std::size_t r = 1; r < n && sel.size() < k; ++r) {
      bool ok = true;
      for (std::size_t s : sel)
        if (D[r][s] < t - 1e-12) {
          ok = false;
          break;
        }
      if (ok)
        sel.push_back(r);
    }
    return sel.size() >= k;
  };

  // Binary search: largest feasible threshold
  std::size_t lo = 0, hi = thresholds.size() - 1;
  while (lo < hi) {
    std::size_t mid = lo + (hi - lo + 1) / 2;
    if (is_feasible(thresholds[mid]))
      lo = mid;
    else
      hi = mid - 1;
  }
  double opt_t = thresholds[lo];

  // Reconstruct
  std::vector<std::size_t> sel_idx;
  sel_idx.reserve(k);
  sel_idx.push_back(0);
  for (std::size_t r = 1; r < n && sel_idx.size() < k; ++r) {
    bool ok = true;
    for (std::size_t s : sel_idx)
      if (D[r][s] < opt_t - 1e-12) {
        ok = false;
        break;
      }
    if (ok)
      sel_idx.push_back(r);
  }

  PointSet<T> subset;
  subset.reserve(k);
  for (std::size_t idx : sel_idx)
    subset.push_back(B[idx]);

  double actual_unif = uniformity(subset, dist);
  return {actual_unif, std::move(subset)};
}

/// @brief Find the k-element subset that minimizes coverage error via
///        binary search on threshold distances + greedy set cover.
///
/// Exact for m=2 (sorted biobjective nondominated points).
/// Complexity: O(n² log n)
///
/// @param B     Non-dominated set (must be biobjective, m=2).
/// @param k     Desired cardinality.
/// @param dist  Distance function (default: Euclidean).
/// @return      RepresentationResult with optimal coverage and subset.
template <typename T = double, typename DistanceFn = EuclideanDistance>
[[nodiscard]] inline RepresentationResult<T>
thresh_min_coverage(PointSet<T> B, std::size_t k,
                    DistanceFn dist = DistanceFn{}) {
  B = detail::prepare_biobjective(std::move(B));
  const std::size_t n = B.size();
  detail::validate_k(k, n);

  if (k >= n)
    return {0.0, B};

  // Precompute all pairwise distances
  std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      if (i != j)
        D[i][j] = dist(B[i], B[j]);

  // Collect unique distances as candidate thresholds
  std::vector<double> thresholds;
  thresholds.reserve(n * (n - 1) / 2);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 1; j < n; ++j)
      thresholds.push_back(D[i][j]);
  std::sort(thresholds.begin(), thresholds.end());
  thresholds.erase(
      std::unique(thresholds.begin(), thresholds.end(),
                  [](double a, double b) { return std::abs(a - b) < 1e-12; }),
      thresholds.end());

  // Greedy feasibility: can we cover all points with k centers within t?
  auto is_feasible = [&](double t) -> bool {
    std::vector<bool> covered(n, false);
    for (std::size_t c = 0; c < k; ++c) {
      std::size_t best_r = 0, best_count = 0;
      for (std::size_t r = 0; r < n; ++r) {
        std::size_t cnt = 0;
        for (std::size_t b = 0; b < n; ++b)
          if (!covered[b] && D[b][r] <= t + 1e-12)
            ++cnt;
        if (cnt > best_count) {
          best_count = cnt;
          best_r = r;
        }
      }
      if (best_count == 0)
        break;
      for (std::size_t b = 0; b < n; ++b)
        if (D[b][best_r] <= t + 1e-12)
          covered[b] = true;
    }
    for (bool c : covered)
      if (!c)
        return false;
    return true;
  };

  // Binary search: smallest feasible threshold
  std::size_t lo = 0, hi = thresholds.size() - 1;
  while (lo < hi) {
    std::size_t mid = lo + (hi - lo) / 2;
    if (is_feasible(thresholds[mid]))
      hi = mid;
    else
      lo = mid + 1;
  }
  double opt_t = thresholds[lo];

  // Reconstruct
  std::vector<bool> selected(n, false), covered(n, false);
  PointSet<T> subset;
  subset.reserve(k);
  for (std::size_t c = 0; c < k; ++c) {
    std::size_t best_r = 0, best_count = 0;
    for (std::size_t r = 0; r < n; ++r) {
      if (selected[r])
        continue;
      std::size_t cnt = 0;
      for (std::size_t b = 0; b < n; ++b)
        if (!covered[b] && D[b][r] <= opt_t + 1e-12)
          ++cnt;
      if (cnt > best_count) {
        best_count = cnt;
        best_r = r;
      }
    }
    selected[best_r] = true;
    subset.push_back(B[best_r]);
    for (std::size_t b = 0; b < n; ++b)
      if (D[b][best_r] <= opt_t + 1e-12)
        covered[b] = true;
  }

  double actual_cov = coverage_error(B, subset, dist);
  return {actual_cov, std::move(subset)};
}

/// @brief Find the k-element subset that minimizes the ε-indicator via
///        binary search on threshold ε-ratios + greedy set cover.
///
/// Exact for m=2 (sorted biobjective nondominated points).
/// Complexity: O(n² log n)
///
/// @param B     Non-dominated set (must be biobjective, m=2, all positive).
/// @param k     Desired cardinality.
/// @param sense optimization sense.
/// @return      RepresentationResult with optimal ε-indicator and subset.
template <typename T = double>
[[nodiscard]] inline RepresentationResult<T>
thresh_min_epsilon(PointSet<T> B, std::size_t k,
                   Sense sense = Sense::Maximize) {
  B = detail::prepare_biobjective(std::move(B));
  const std::size_t n = B.size();
  detail::validate_k(k, n);

  if (k >= n)
    return {1.0, B};

  // Precompute pairwise ε-ratios: E[b][r] = ε(r, b, sense)
  std::vector<std::vector<double>> E(n, std::vector<double>(n, 0.0));
  for (std::size_t b = 0; b < n; ++b)
    for (std::size_t r = 0; r < n; ++r)
      E[b][r] = epsilon_ratio(B[r], B[b], sense);

  // Collect unique ε values as candidate thresholds
  std::vector<double> thresholds;
  thresholds.reserve(n * n);
  for (std::size_t b = 0; b < n; ++b)
    for (std::size_t r = 0; r < n; ++r)
      thresholds.push_back(E[b][r]);
  std::sort(thresholds.begin(), thresholds.end());
  thresholds.erase(
      std::unique(thresholds.begin(), thresholds.end(),
                  [](double a, double b) { return std::abs(a - b) < 1e-12; }),
      thresholds.end());

  // Greedy feasibility: can we cover all points with k centers within ε ≤ t?
  auto is_feasible = [&](double t) -> bool {
    std::vector<bool> covered(n, false);
    for (std::size_t c = 0; c < k; ++c) {
      std::size_t best_r = 0, best_count = 0;
      for (std::size_t r = 0; r < n; ++r) {
        std::size_t cnt = 0;
        for (std::size_t b = 0; b < n; ++b)
          if (!covered[b] && E[b][r] <= t + 1e-12)
            ++cnt;
        if (cnt > best_count) {
          best_count = cnt;
          best_r = r;
        }
      }
      if (best_count == 0)
        break;
      for (std::size_t b = 0; b < n; ++b)
        if (E[b][best_r] <= t + 1e-12)
          covered[b] = true;
    }
    for (bool c : covered)
      if (!c)
        return false;
    return true;
  };

  // Binary search: smallest feasible threshold
  std::size_t lo = 0, hi = thresholds.size() - 1;
  while (lo < hi) {
    std::size_t mid = lo + (hi - lo) / 2;
    if (is_feasible(thresholds[mid]))
      hi = mid;
    else
      lo = mid + 1;
  }
  double opt_t = thresholds[lo];

  // Reconstruct
  std::vector<bool> selected(n, false), covered(n, false);
  PointSet<T> subset;
  subset.reserve(k);
  for (std::size_t c = 0; c < k; ++c) {
    std::size_t best_r = 0, best_count = 0;
    for (std::size_t r = 0; r < n; ++r) {
      if (selected[r])
        continue;
      std::size_t cnt = 0;
      for (std::size_t b = 0; b < n; ++b)
        if (!covered[b] && E[b][r] <= opt_t + 1e-12)
          ++cnt;
      if (cnt > best_count) {
        best_count = cnt;
        best_r = r;
      }
    }
    selected[best_r] = true;
    subset.push_back(B[best_r]);
    for (std::size_t b = 0; b < n; ++b)
      if (E[b][best_r] <= opt_t + 1e-12)
        covered[b] = true;
  }

  double actual_eps = epsilon_indicator(B, subset, sense);
  return {actual_eps, std::move(subset)};
}

// ===================================================================
// 7–10. Combined DP algorithms (Vaz et al. §6)
// ===================================================================

/// @brief Find the k-element subset that minimizes coverage subject to
///        a uniformity constraint (or vice versa).
///
/// Implements the combined Coverage–Uniformity algorithm from §6.1.
///
/// @param B     Non-dominated set (must be biobjective, m=2).
/// @param k     Desired cardinality of representation.
/// @param dist  Distance function (default: Euclidean).
/// @return      RepresentationResult with optimal value and subset.
template <typename T = double, typename DistanceFn = EuclideanDistance>
[[nodiscard]] inline RepresentationResult<T>
dp_coverage_uniformity(PointSet<T> B, std::size_t k,
                       DistanceFn dist = DistanceFn{}) {
  (void)B;
  (void)k;
  (void)dist;
  throw std::logic_error(
      "moqm::dp_coverage_uniformity: not yet implemented (§6.1).");
}

/// @brief Find the k-element subset that minimizes ε-indicator subject to
///        a uniformity constraint (or vice versa).
///
/// Implements the combined ε-Indicator–Uniformity algorithm from §6.2.
///
/// @param B     Non-dominated set (must be biobjective, m=2, all positive).
/// @param k     Desired cardinality of representation.
/// @param sense optimization sense (Maximize or Minimize).
/// @return      RepresentationResult with optimal value and subset.
template <typename T = double>
[[nodiscard]] inline RepresentationResult<T>
dp_epsilon_uniformity(PointSet<T> B, std::size_t k,
                      Sense sense = Sense::Maximize) {
  (void)B;
  (void)k;
  (void)sense;
  throw std::logic_error(
      "moqm::dp_epsilon_uniformity: not yet implemented (§6.2).");
}

/// @brief Find the k-element subset that minimizes coverage subject to
///        an ε-indicator constraint (or vice versa).
///
/// Implements the combined Coverage–ε-Indicator algorithm from §6.3.
///
/// @param B     Non-dominated set (must be biobjective, m=2, all positive).
/// @param k     Desired cardinality of representation.
/// @param dist  Distance function (default: Euclidean).
/// @param sense optimization sense (Maximize or Minimize).
/// @return      RepresentationResult with optimal value and subset.
template <typename T = double, typename DistanceFn = EuclideanDistance>
[[nodiscard]] inline RepresentationResult<T>
dp_coverage_epsilon(PointSet<T> B, std::size_t k,
                    DistanceFn dist = DistanceFn{},
                    Sense sense = Sense::Maximize) {
  (void)B;
  (void)k;
  (void)dist;
  (void)sense;
  throw std::logic_error(
      "moqm::dp_coverage_epsilon: not yet implemented (§6.3).");
}

/// @brief Find the k-element subset that simultaneously optimizes
///        coverage, ε-indicator, and uniformity.
///
/// Implements the combined Coverage–ε-Indicator–Uniformity algorithm
/// from §6.4.
///
/// @param B     Non-dominated set (must be biobjective, m=2, all positive).
/// @param k     Desired cardinality of representation.
/// @param dist  Distance function (default: Euclidean).
/// @param sense optimization sense (Maximize or Minimize).
/// @return      RepresentationResult with optimal value and subset.
template <typename T = double, typename DistanceFn = EuclideanDistance>
[[nodiscard]] inline RepresentationResult<T>
dp_coverage_epsilon_uniformity(PointSet<T> B, std::size_t k,
                               DistanceFn dist = DistanceFn{},
                               Sense sense = Sense::Maximize) {
  (void)B;
  (void)k;
  (void)dist;
  (void)sense;
  throw std::logic_error(
      "moqm::dp_coverage_epsilon_uniformity: not yet implemented (§6.4).");
}

} // namespace moqm
