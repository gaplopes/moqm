#pragma once

/// @file moqm.hpp
/// @brief Umbrella header for the moqm (Multiobjective Quality Metrics)
///        header-only library.
///
/// Including this single header provides access to all moqm functionality.
///
/// Dependency-free modules (no GLPK required):
///   - moqm/point.hpp               Point type and utilities
///   - moqm/distance.hpp            Distance functions
///   - moqm/indicators.hpp          Quality indicator evaluation
///   - moqm/representation.hpp      DP and threshold representation solvers
///
/// GLPK-dependent modules (require linking with -lglpk):
///   - moqm/classification.hpp      Supported/extreme point identification

#include "moqm/classification.hpp"
#include "moqm/distance.hpp"
#include "moqm/indicators.hpp"
#include "moqm/point.hpp"
#include "moqm/representation.hpp"
