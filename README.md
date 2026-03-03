# moqm — Multiobjective Quality Metrics

> **Header-only C++17 library** for evaluating the quality of representations of nondominated sets in multiobjective discrete optimization problems.

## Overview

In multiobjective optimization, the nondominated (Pareto) set can be very large. A common strategy is to select a small *representative* subset that preserves the essential structure of the full set. This library provides:

1. **Quality indicators** — mathematically rigorous functions that measure how well a representation captures the nondominated set.
2. **Representation solvers** — algorithms that find *optimal* representative subsets of a given cardinality with respect to a specified indicator.
3. **Point classification** — LP-based identification of supported and extreme supported nondominated points.

### Theoretical Foundations

The library implements quality metrics and algorithms from two key papers:

- **Sayın (2024)** — *"Supported nondominated points as a representation of the nondominated set: An empirical analysis."* Journal of Multi-Criteria Decision Analysis.
  - Coverage Error (CE), Median Error (ME), Hypervolume Ratio (HVR), Range Ratio (RR)
  - LP formulations S(y^k) and E(y^k) for identifying supported and extreme supported nondominated points

- **Vaz, Paquete, Fonseca, Klamroth, Stiglmayr (2015)** — *"Representation of the non-dominated set in biobjective discrete optimization."* Computers & Operations Research.
  - Uniformity (I_U), Coverage (I_C), ε-indicator (I_ε)
  - O(kn) dynamic programming algorithms for optimal representations (biobjective case)
  - Threshold-based algorithms for optimal representations (biobjective case)

## Features

| Feature | Description | Dimension |
|---|---|---|
| Coverage Error (CE) | Worst-case distance from Y_N to R | Any m |
| Median Error (ME) | Median of per-point errors | Any m |
| Range Ratio (RR) | Average per-objective range coverage | Any m |
| Hypervolume Ratio (HVR) | HV(R) / HV(Y_N) | m = 2 only |
| Uniformity (I_U) | Min pairwise distance in R | Any m |
| ε-Indicator (I_ε) | Multiplicative approximation ratio | Any m |
| DP Uniformity Solver | k-subset maximizing I_U | m = 2 only |
| DP Coverage Solver | k-subset minimizing I_C | m = 2 only |
| DP ε-Indicator Solver | k-subset minimizing I_ε | m = 2 only |
| Threshold Uniformity Solver | k-subset maximizing I_U | m = 2 only |
| Threshold Coverage Solver | k-subset minimizing I_C | m = 2 only |
| Threshold ε-Indicator Solver | k-subset minimizing I_ε | m = 2 only |
| Point Classification | Supported / extreme / unsupported | Any m |

There are also stubs for the following features:

| Feature | Description | Dimension |
|---|---|---|
| DP Coverage–Uniformity | **Stub** — not yet implemented | m = 2 only |
| DP ε-Indicator–Uniformity | **Stub** — not yet implemented | m = 2 only |
| DP Coverage–ε-Indicator | **Stub** — not yet implemented | m = 2 only |
| DP Coverage–ε-Ind.–Unif. | **Stub** — not yet implemented | m = 2 only |

### DP vs Threshold Solvers

The library offers two families of representation solvers:

- **DP solvers** (`dp_max_uniformity`, `dp_min_coverage`, `dp_min_epsilon`) — *exact*
  O(kn) algorithms that exploit the monotonicity structure of sorted biobjective
  nondominated sets (Vaz et al., 2015).  Valid only for **m = 2**.

- **Threshold solvers** (`thresh_max_uniformity`, `thresh_min_coverage`,
  `thresh_min_epsilon`) — exact algorithms that binary-search over candidate
  threshold values and use greedy selection.  The greedy is exact for **m = 2**
  because the 1D sorted ordering of biobjective nondominated points guarantees
  distance monotonicity (Proposition 2.1).  They are O(n² log n) — slower than
  the O(kn) DP solvers — but serve as **independent cross-validation**.

## Library Architecture

```
include/
├── moqm.hpp                      # Umbrella header (includes everything)
└── moqm/
    ├── point.hpp                  # Point<T> type, dominance, sorting
    ├── distance.hpp               # Distance functions and functors
    ├── indicators.hpp             # Quality indicator evaluation
    ├── representation.hpp         # DP + threshold solvers (m=2) [NO DEPS]
    └── classification.hpp         # LP-based classification [GLPK]
```

### Dependencies

| Module | External Dependencies |
|---|---|
| `point.hpp` | None (C++17 STL only) |
| `distance.hpp` | None |
| `indicators.hpp` | None |
| `representation.hpp` | None |
| `classification.hpp` | **GLPK** |

The only module that requires GLPK is `classification.hpp` (for LP-based supported/extreme point identification).  All other modules — including the DP and threshold solvers — are dependency-free.

## Building & Testing

### With CMake

```bash
mkdir build && cd build
cmake ..
cmake --build .
ctest # run moqm library tests
```

The library is available as the `moqm` CMake target:
```cmake
target_link_libraries(your_target PRIVATE moqm)
```

### Standalone (no CMake, no GLPK)

Since moqm is header-only, you can simply copy the `include/moqm/` directory
into your project and compile directly:

```bash
# Without GLPK (all modules except classification)
g++ -std=c++17 -I include your_code.cpp -o your_program

# With GLPK (classification module)
g++ -std=c++17 -I include your_code.cpp -lglpk -o your_program

# Run the library tests
g++ -std=c++17 -I include tests/test_moqm.cpp -o test_moqm && ./test_moqm
```

## Templated `Point<T>`

The `Point` class is templated on the value type:

```cpp
Point<double> p1({1.0, 2.0, 3.0});  // default (double)
Point<int>    p2({1, 2, 3});         // integer points
Point<float>  p3({1.0f, 2.0f});      // float points

// Backward-compatible aliases
using PointD    = Point<double>;
using PointSetD = PointSet<double>;
```

All indicator functions, distance functions, and solvers are templated to work with `Point<T>` for any numeric type `T`.

## Unified API with `Sense` Parameter

Functions that depend on the optimization direction take a `Sense` parameter instead of separate `_max`/`_min` variants:

```cpp
// ε-indicator (unified)
double eps = epsilon_indicator(Y_N, R, Sense::Maximize);
double eps = epsilon_indicator(Y_N, R, Sense::Minimize);

// Hypervolume 2D (unified)
double hv = detail::hypervolume_2d(pts, ref, Sense::Maximize);

// ε-ratio (unified)
double er = epsilon_ratio(r, b, Sense::Maximize);
```

## Mathematical Definitions

### Quality Indicators

**Coverage Error**:
$$\text{CE}(R) = \max_{y \in Y_N} \min_{r \in R} d(y, r)$$

where $d$ is the weighted Tchebycheff distance with $w_j = 1/(\max_{Y_N} y_j - \min_{Y_N} y_j)$.

**Median Error**:
$$\text{ME}(R) = \text{median}\{\min_{r \in R} d(y, r) : y \in Y_N\}$$

**Range Ratio**:
$$\text{RR}(R) = \frac{1}{m} \sum_{j=1}^{m} \frac{\max_{r \in R} r_j - \min_{r \in R} r_j}{\max_{y \in Y_N} y_j - \min_{y \in Y_N} y_j}$$

**Hypervolume Ratio**:
$$\text{HVR}(R) = \frac{HV(R)}{HV(Y_N)}$$

**Uniformity**:
$$I_U(R) = \min_{r_i \neq r_j \in R} \|r_i - r_j\|$$

**ε-Indicator**:
$$I_\varepsilon(R, B) = \max_{b \in B} \min_{r \in R} \max_i \frac{b_i}{r_i}$$

### Biobjective DP Algorithms (Vaz et al., 2015)

All three algorithms share the same DP structure over a k × n matrix T, exploiting the monotonicity of inter-point distances for sorted biobjective nondominated points (Proposition 2.1).

**Uniformity DP** (§3.1):
```
Base:      T(1, j) = +∞
Recursion: T(i, j) = max_{j<ℓ≤n-i+2} min(‖b_j − b_ℓ‖, T(i−1, ℓ))
Result:    max_{1≤j≤n−k+1} T(k, j)
```

**Coverage DP** (§4.1 with m/ℓ-improvements):
```
Base:       T(1, j) = ‖b_j − b_n‖
δ_{j,ℓ}:   max_{j≤m≤ℓ} min(‖b_j − b_m‖, ‖b_m − b_ℓ‖)
Recursion:  T(i, j) = min_{j<ℓ≤n-i+2} max(δ_{j,ℓ}, T(i−1, ℓ))
Correction: T̄(k, j) = max(‖b_1 − b_j‖, T(k, j))
Result:     min_{1≤j≤n−k+1} T̄(k, j)
```

**ε-Indicator DP** (§5.1):
```
Same structure as Coverage DP but with ε-ratio replacing norms:
δ_{j,ℓ} = max_{j≤m≤ℓ} min(ε(b_j, b_m), ε(b_ℓ, b_m))
Correction: T̄(k, j) = max(ε(b_j, b_1), T(k, j))
```

> **Important:** These O(kn) algorithms are valid only for **biobjective (m=2)** problems. For m ≥ 3, the k-center and k-dispersion problems are NP-hard.

### Threshold Algorithms (Cross-Validation)

All three threshold solvers follow the same pattern:

1. **Precompute** all pairwise distances (or ε-ratios) — O(n²).
2. **Collect** all unique values as candidate thresholds.
3. **Binary search** on thresholds: for each candidate, run a greedy check.
4. **Reconstruct** the solution at the optimal threshold.

For **m = 2**, the greedy selection is exact because the sorted staircase
ordering of biobjective nondominated points guarantees that inter-point
distances are monotone (Proposition 2.1 of Vaz et al.).  This makes the
threshold solvers a valid **cross-validation** tool: they should produce
optimal values matching the DP solvers.

### Point Classification (Sayın, 2024)

**S(y^k) — Supported point test:**
Find λ ≥ 0, Σλ_i = 1 such that λᵀy^k ≤ λᵀy^j for all j. If feasible, y^k ∈ Y_SN.

**E(y^k) — Extreme supported point test:**
min α_k s.t. Σα_j · y^j = y^k, Σα_j = 1, α ≥ 0. If α_k* = 1, then y^k ∈ Y_ESN.

## Complexity Analysis

| Operation | Time Complexity | Space |
|---|---|---|
| Coverage Error | O(\|Y_N\| · \|R\| · m) | O(1) |
| Median Error | O(\|Y_N\| · \|R\| · m + \|Y_N\| log \|Y_N\|) | O(\|Y_N\|) |
| Range Ratio | O((\|Y_N\| + \|R\|) · m) | O(m) |
| Hypervolume (2D) | O(n log n) | O(n) |
| Uniformity | O(\|R\|² · m) | O(1) |
| ε-Indicator | O(\|Y_N\| · \|R\| · m) | O(1) |
| DP Uniformity | O(kn + n log n) | O(kn) |
| DP Coverage | O(kn + n log n) | O(kn) |
| DP ε-Indicator | O(kn + n log n) | O(kn) |
| Threshold solvers | O(n² log n) | O(n²) |
| is_supported | O(Mp) per point (LP) | O(Mp) |

## Usage Examples

### Evaluating Quality Indicators

```cpp
#include <moqm/indicators.hpp>
#include <moqm/distance.hpp>
#include <moqm/point.hpp>

#include <iostream>

int main() {
    using namespace moqm;

    // Example from Sayın (2024), Figure 1:
    PointSet<> Y_N = {
        {1, 8}, {2, 7}, {3, 4}, {5, 3}, {6, 2.5}, {8, 1}
    };
    PointSet<> R = {{1, 8}, {3, 4}, {8, 1}};  // Y_SN = Y_ESN

    // Coverage Error (weighted Tchebycheff, auto-computed weights)
    double ce = coverage_error(Y_N, R);
    std::cout << "CE(R)  = " << ce << "\n";     // ≈ 0.2857

    // Median Error
    double me = median_error(Y_N, R);
    std::cout << "ME(R)  = " << me << "\n";     // ≈ 0.0714

    // Range Ratio
    double rr = range_ratio(Y_N, R);
    std::cout << "RR(R)  = " << rr << "\n";     // = 1.0

    // Uniformity (Euclidean)
    double iu = uniformity(R);
    std::cout << "I_U(R) = " << iu << "\n";

    // ε-Indicator (unified with Sense parameter)
    double ie = epsilon_indicator(Y_N, R, Sense::Maximize);
    std::cout << "I_ε(R) = " << ie << "\n";

    // Hypervolume Ratio
    Point<> ref = {0, 0};  // reference point below all
    double hvr = hypervolume_ratio(Y_N, R, ref, Sense::Maximize);
    std::cout << "HVR(R) = " << hvr << "\n";

    // Using custom discrete distance (Chebyshev)
    double ce_cheb = coverage_error(Y_N, R, ChebyshevDistance{});
    std::cout << "CE_cheb = " << ce_cheb << "\n";

    // Auto-calculating weights for Weighted Tchebycheff based on Y_N
    double ce_tch = coverage_error(Y_N, R, WeightedTchebycheffDistance(Y_N));
    std::cout << "CE_tch = " << ce_tch << "\n";

    return 0;
}
```

### Finding Optimal Representations (Biobjective DP)

```cpp
#include <moqm/representation.hpp>
#include <iostream>

int main() {
    using namespace moqm;

    // Biobjective nondominated set (m = 2)
    PointSet<> B = {
        {1, 10}, {2, 8}, {3, 7}, {4, 5},
        {6, 4}, {7, 3}, {9, 2}, {10, 1}
    };
    std::size_t k = 3;  // select 3 representatives

    // Maximize uniformity (exact, O(kn)) — uses default EuclideanDistance
    auto [u_val, u_subset] = dp_max_uniformity(B, k);
    std::cout << "Optimal uniformity: " << u_val << "\n";
    for (const auto& p : u_subset) std::cout << p.to_string() << " ";
    std::cout << "\n";

    // Minimize coverage (exact, O(kn)) — changing default to WeightedTchebycheffDistance
    auto [c_val, c_subset] = dp_min_coverage(B, k, WeightedTchebycheffDistance(B));
    std::cout << "Optimal coverage: " << c_val << "\n";

    // Minimize ε-indicator (exact, O(kn))
    auto [e_val, e_subset] = dp_min_epsilon(B, k, Sense::Maximize);
    std::cout << "Optimal ε-indicator: " << e_val << "\n";

    return 0;
}
```

### Threshold Solvers (Cross-Validation)

```cpp
#include <moqm/representation.hpp>
#include <iostream>

int main() {
    using namespace moqm;

    // Cross-validate DP results with threshold algorithms (m=2)
    PointSet<> B = {
        {1, 10}, {2, 8}, {3, 7}, {4, 5},
        {6, 4}, {7, 3}, {9, 2}, {10, 1}
    };
    std::size_t k = 3;

    // Threshold solvers (exact for m=2, O(n² log n))
    auto [u_val, u_sub] = thresh_max_uniformity(B, k);
    auto [c_val, c_sub] = thresh_min_coverage(B, k);
    auto [e_val, e_sub] = thresh_min_epsilon(B, k, Sense::Maximize);

    std::cout << "Uniformity: " << u_val << "\n";
    std::cout << "Coverage:   " << c_val << "\n";
    std::cout << "ε-Indicator:" << e_val << "\n";

    return 0;
}
```

### Classifying Nondominated Points

```cpp
#include <moqm/classification.hpp>
#include <iostream>

int main() {
    using namespace moqm;

    PointSet<> Y_N = {
        {1, 8}, {2, 7}, {3, 4}, {5, 3}, {6, 2.5}, {8, 1}
    };

    auto cls = classify(Y_N, Sense::Minimize);

    std::cout << "|Y_SN|  = " << cls.supported.size() << "\n";
    std::cout << "|Y_ESN| = " << cls.extreme_supported.size() << "\n";
    std::cout << "|Y_USN| = " << cls.unsupported.size() << "\n";

    return 0;
}
```

## Distance Metrics

Both indicators and representation solvers require a method to measure distance between points. If a custom distance is not provided, the library defaults to the exact mathematical definitions established in the source literature:
- `uniformity` estimators (`dp_max_uniformity`, `thresh_max_uniformity`): **EuclideanDistance**
- `coverage_error` estimators (`dp_min_coverage`, `thresh_min_coverage`, `median_error`): **WeightedTchebycheffDistance**
- `epsilon_indicator` estimators (`dp_min_epsilon`, `thresh_min_epsilon`): uses a directional **EpsilonRatio** internally.

You can override these defaults by passing a custom functor as the final argument. The library natively provides the following ready-to-use metrics:
- `EuclideanDistance()`
- `ChebyshevDistance()`
- `LpNormDistance(double p)`
- `EpsilonRatio(Sense s)`
- `WeightedTchebycheffDistance(std::vector<double> w)` or `WeightedTchebycheffDistance(const PointSet<T>& reference_set)`

### Examples
You can easily pass these distance metrics directly into your indicators and representation solvers.

**Using Euclidean Distance (Default for Uniformity):**
```cpp
#include <moqm/indicators.hpp>
#include <moqm/distance.hpp>

// Explicitly using Euclidean distance
double iu = uniformity(R, EuclideanDistance{});
```

**Using Weighted Tchebycheff (Default for Coverage):**
The library offers a constructor for the Weighted Tchebycheff metric. By passing the complete reference set $Y_N$ to the constructor, the library **automatically** calculates the $\max$ and $\min$ boundary values across all dimensions to generate the appropriate normalized weights dynamically:

```cpp
#include <moqm/distance.hpp>

// Automatically calculates w_j = 1 / (max_j - min_j) across the set B
WeightedTchebycheffDistance dist_tch(B);

// The functor is then ready to be plugged into any solver!
auto result = dp_min_coverage(B, k, dist_tch);
```

**Using an Explicit L-p Norm:**
```cpp
#include <moqm/distance.hpp>

// Use L-3 Norm
LpNormDistance dist_lp3(3.0);
double ce = coverage_error(Y_N, R, dist_lp3);
```

## Namespace

All symbols are in the `moqm` namespace:

```cpp
using namespace moqm;              // or
moqm::Point<> p({1.0, 2.0});      // qualified access
moqm::Point<int> q({1, 2, 3});    // integer points
```

## Limitations

1. **Biobjective DP solvers** require exactly m = 2 objectives. They throw `std::invalid_argument` if applied to higher-dimensional point sets.
2. **Combined DP solvers** (§6.1–6.4) are declared but not yet implemented. Calling them throws `std::logic_error`.
3. **Threshold solvers** are exact for m=2 (exploiting the sorted staircase property) and serve as cross-validation for the DP solvers. They are O(n² log n), slower than O(kn) DP.
4. **Hypervolume** uses an exact sweep-line for 2D. For m ≥ 3, it is not implemented yet.
5. **ε-indicator** requires all point components to be strictly positive. `epsilon_ratio` throws `std::invalid_argument` if a denominator component is ≤ 0.
6. **GLPK dependency** is required only for `classification.hpp`.

## Contributing

Contributions are welcome! If you have bug fixes, new problem implementations, additional strategies, or other improvements, please fork the repository and open a pull request. For major changes, consider opening an issue first to discuss the approach.

## Citation

If you use this library in your research, please cite the original papers and this implementation as follows:

```bibtex
@software{moqm,
  author = {Lopes, Gon{\c{c}}alo},
  title = {{moqm}: A {C++} Library for Multiobjective Quality Metrics of Nondominated Set Representations},
  year = {2026},
  url = {https://github.com/gaplopes/moqm}
}
```

## References

1. Sayın, S. (2024). Supported nondominated points as a representation of the nondominated set: An empirical analysis. *Journal of Multi-Criteria Decision Analysis*, 31, e1829.
2. Vaz, D., Paquete, L., Fonseca, C. M., Klamroth, K., & Stiglmayr, M. (2015). Representation of the non-dominated set in biobjective discrete optimization. *Computers & Operations Research*, 63, 172–186.

## License

See [LICENSE](LICENSE) for details.
