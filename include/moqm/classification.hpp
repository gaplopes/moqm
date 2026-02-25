#pragma once

/// @file classification.hpp
/// @brief Identification of supported and extreme supported nondominated
///        points using linear programming (GLPK).
///
/// Implements the formulations S(y^k) and E(y^k) from:
///   Sayın (2024). "Supported nondominated points as a representation of
///   the nondominated set: An empirical analysis."
///
/// REQUIRES: GLPK library linked to the project.
///
/// Part of the moqm (Multiobjective Quality Metrics) header-only library.

#include <glpk.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "point.hpp"

namespace moqm {

namespace detail {

/// @brief RAII wrapper for GLPK problem objects.
struct GlpkProblemDeleter {
  void operator()(glp_prob *p) const {
    if (p)
      glp_delete_prob(p);
  }
};
using GlpkProbPtr = std::unique_ptr<glp_prob, GlpkProblemDeleter>;

/// @brief Create a GLPK problem and return an RAII handle.
[[nodiscard]] inline GlpkProbPtr make_glpk_prob(const char *name) {
  glp_prob *lp = glp_create_prob();
  if (!lp)
    throw std::runtime_error("moqm: failed to create GLPK problem.");
  glp_set_prob_name(lp, name);
  return GlpkProbPtr(lp);
}

} // namespace detail

// ===================================================================
// Classification result
// ===================================================================

/// @brief Classification of a nondominated set.
template <typename T = double> struct Classification {
  PointSet<T> supported;
  PointSet<T> extreme_supported;
  PointSet<T> unsupported;
};

// ===================================================================
// S(y^k): Test if a point is supported nondominated
// ===================================================================

/// @brief Check if point y_k is a supported nondominated point.
///
/// Solves the LP formulation S(y^k) from Sayın (2024):
///   max Σ λ_i
///   s.t. Σ λ_i · (y^k_i - y^j_i) ≤ 0  for minimization (or reversed)
///        Σ λ_i = 1
///        λ_i ≥ 0
///
/// @param y_k   Candidate point.
/// @param Y_N   Complete nondominated set.
/// @param sense optimization sense.
/// @return      True if y_k is supported.
template <typename T>
[[nodiscard]] inline bool is_supported(const Point<T> &y_k,
                                       const PointSet<T> &Y_N, Sense sense) {
  const int p = static_cast<int>(y_k.dim());
  const int M = static_cast<int>(Y_N.size());
  if (M == 0)
    return false;

  auto lp = detail::make_glpk_prob("is_supported");
  glp_set_obj_dir(lp.get(), GLP_MIN);

  // Columns: λ_1 ... λ_p
  glp_add_cols(lp.get(), p);
  for (int i = 1; i <= p; ++i) {
    glp_set_col_bnds(lp.get(), i, GLP_LO, 0.0, 0.0);
    glp_set_obj_coef(lp.get(), i, 0.0);
  }

  // Rows: M inequality constraints + 1 equality
  glp_add_rows(lp.get(), M + 1);
  for (int j = 0; j < M; ++j) {
    glp_set_row_bnds(lp.get(), j + 1, GLP_UP, 0.0, 0.0);
  }
  glp_set_row_bnds(lp.get(), M + 1, GLP_FX, 1.0, 1.0);

  // Fill constraint matrix
  const int nnz = M * p + p;
  std::vector<int> ia(1 + nnz), ja(1 + nnz);
  std::vector<double> ar(1 + nnz);
  int idx = 1;

  for (int j = 0; j < M; ++j) {
    for (int i = 0; i < p; ++i) {
      ia[idx] = j + 1;
      ja[idx] = i + 1;
      auto sj = static_cast<std::size_t>(j);
      auto si = static_cast<std::size_t>(i);
      if (sense == Sense::Maximize) {
        ar[idx] =
            static_cast<double>(Y_N[sj][si]) - static_cast<double>(y_k[si]);
      } else {
        ar[idx] =
            static_cast<double>(y_k[si]) - static_cast<double>(Y_N[sj][si]);
      }
      ++idx;
    }
  }
  for (int i = 0; i < p; ++i) {
    ia[idx] = M + 1;
    ja[idx] = i + 1;
    ar[idx] = 1.0;
    ++idx;
  }

  glp_load_matrix(lp.get(), nnz, ia.data(), ja.data(), ar.data());

  glp_smcp smcp;
  glp_init_smcp(&smcp);
  smcp.msg_lev = GLP_MSG_OFF;
  smcp.presolve = GLP_ON;
  glp_simplex(lp.get(), &smcp);

  int status = glp_get_status(lp.get());
  return (status == GLP_OPT || status == GLP_FEAS);
}

// ===================================================================
// E(y^k): Test if a point is extreme supported nondominated
// ===================================================================

/// @brief Check if a supported point y_k is extreme supported.
///
/// Solves the LP formulation E(y^k) from Sayın (2024):
///   min α_k
///   s.t. Σ α_j · y^j_i = y^k_i  for each objective i
///        Σ α_j = 1
///        α_j ≥ 0
///
/// @param y_k   Candidate supported point.
/// @param Y_SN  Set of all supported nondominated points.
/// @return      True if y_k is extreme supported (α_k* = 1).
template <typename T>
[[nodiscard]] inline bool is_extreme_supported(const Point<T> &y_k,
                                               const PointSet<T> &Y_SN) {
  const int p = static_cast<int>(y_k.dim());
  const int N = static_cast<int>(Y_SN.size());
  if (N == 0)
    return false;

  // Find the index of y_k in Y_SN.
  int k_idx = -1;
  for (int j = 0; j < N; ++j) {
    if (Y_SN[static_cast<std::size_t>(j)] == y_k) {
      k_idx = j;
      break;
    }
  }
  if (k_idx < 0)
    return false;

  auto lp = detail::make_glpk_prob("is_extreme_supported");
  glp_set_obj_dir(lp.get(), GLP_MIN);

  // Columns: α_1 ... α_N
  glp_add_cols(lp.get(), N);
  for (int j = 1; j <= N; ++j) {
    glp_set_col_bnds(lp.get(), j, GLP_LO, 0.0, 0.0);
    glp_set_obj_coef(lp.get(), j, (j == k_idx + 1) ? 1.0 : 0.0);
  }

  // Rows: p equality constraints + 1 sum constraint
  glp_add_rows(lp.get(), p + 1);
  for (int i = 0; i < p; ++i) {
    double val = static_cast<double>(y_k[static_cast<std::size_t>(i)]);
    glp_set_row_bnds(lp.get(), i + 1, GLP_FX, val, val);
  }
  glp_set_row_bnds(lp.get(), p + 1, GLP_FX, 1.0, 1.0);

  // Fill constraint matrix
  const int nnz = p * N + N;
  std::vector<int> ia(1 + nnz), ja(1 + nnz);
  std::vector<double> ar(1 + nnz);
  int idx = 1;

  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < N; ++j) {
      ia[idx] = i + 1;
      ja[idx] = j + 1;
      auto sj = static_cast<std::size_t>(j);
      auto si = static_cast<std::size_t>(i);
      ar[idx] = static_cast<double>(Y_SN[sj][si]);
      ++idx;
    }
  }
  for (int j = 0; j < N; ++j) {
    ia[idx] = p + 1;
    ja[idx] = j + 1;
    ar[idx] = 1.0;
    ++idx;
  }

  glp_load_matrix(lp.get(), nnz, ia.data(), ja.data(), ar.data());

  glp_smcp smcp;
  glp_init_smcp(&smcp);
  smcp.msg_lev = GLP_MSG_OFF;
  smcp.presolve = GLP_ON;
  glp_simplex(lp.get(), &smcp);

  if (glp_get_status(lp.get()) != GLP_OPT)
    return false;
  return std::abs(glp_get_obj_val(lp.get()) - 1.0) < 1e-6;
}

// ===================================================================
// Nondominated set classification
// ===================================================================

/// @brief Classify all points in Y_N as supported, extreme supported,
///        or unsupported.
///
/// Complexity: O(|Y_N|²) LP solves.
///
/// @param Y_N   Complete nondominated set.
/// @param sense optimization sense.
/// @return      Classification with Y_SN, Y_ESN, Y_USN.
template <typename T>
[[nodiscard]] inline Classification<T> classify(const PointSet<T> &Y_N,
                                                Sense sense) {
  Classification<T> result;

  // Step 1: identify supported points.
  for (const auto &y : Y_N) {
    if (is_supported(y, Y_N, sense)) {
      result.supported.push_back(y);
    } else {
      result.unsupported.push_back(y);
    }
  }

  // Step 2: among supported, identify extreme supported.
  for (const auto &y : result.supported) {
    if (is_extreme_supported(y, result.supported)) {
      result.extreme_supported.push_back(y);
    }
  }

  return result;
}

} // namespace moqm
