#include <moqm/indicators.hpp>
#include <moqm/representation.hpp>

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
  using namespace moqm;

  std::cout << "=== moqm library verification ===\n\n";

  // -----------------------------------------------------------------
  // Test data from Sayın (2024), Figure 1
  // -----------------------------------------------------------------
  PointSet<> Y_N = {{1, 8}, {2, 7}, {3, 4}, {5, 3}, {6, 2.5}, {8, 1}};
  PointSet<> R = {{1, 8}, {3, 4}, {8, 1}};

  // --- Coverage Error ---
  double ce = coverage_error(Y_N, R);
  std::cout << "CE(R)  = " << ce << "\n";

  // --- Median Error ---
  double me = median_error(Y_N, R);
  std::cout << "ME(R)  = " << me << "\n";

  // --- Range Ratio ---
  double rr = range_ratio(Y_N, R);
  std::cout << "RR(R)  = " << rr << "\n";
  assert(std::abs(rr - 1.0) < 1e-9 && "RR should be 1.0");

  // --- Uniformity ---
  double iu = uniformity(R);
  std::cout << "I_U(R) = " << iu << "\n";

  // --- ε-Indicator (unified with Sense) ---
  double ie = epsilon_indicator(Y_N, Y_N, Sense::Maximize);
  std::cout << "I_ε(Y_N, Y_N, Max) = " << ie << "\n";
  assert(std::abs(ie - 1.0) < 1e-9 && "ε should be 1.0 when R=B");

  double ie_min = epsilon_indicator(Y_N, Y_N, Sense::Minimize);
  std::cout << "I_ε(Y_N, Y_N, Min) = " << ie_min << "\n";
  assert(std::abs(ie_min - 1.0) < 1e-9 && "ε should be 1.0 when R=B (min)");

  // --- Hypervolume (2D, unified with Sense) ---
  Point<> ref_max = {0, 0};
  double hv_yn = hypervolume(Y_N, ref_max, Sense::Maximize);
  double hv_r = hypervolume(R, ref_max, Sense::Maximize);
  double hvr = hypervolume_ratio(Y_N, R, ref_max, Sense::Maximize);
  std::cout << "HV(Y_N, max) = " << hv_yn << "\n";
  std::cout << "HV(R, max)   = " << hv_r << "\n";
  std::cout << "HVR(R, max)  = " << hvr << "\n";

  // Test minimization HV
  Point<> ref_min = {10, 10};
  double hv_min = hypervolume(Y_N, ref_min, Sense::Minimize);
  std::cout << "HV(Y_N, min) = " << hv_min << "\n";

  // --- Point<int> test (template flexibility) ---
  PointSet<int> int_pts = {{1, 8}, {2, 7}, {3, 4}};
  double ce_int = coverage_error(int_pts, int_pts, EuclideanDistance{});
  std::cout << "\nCE(int pts) = " << ce_int << "\n";
  assert(std::abs(ce_int) < 1e-9 && "CE should be 0 when R=Y_N");

  std::cout << "\n=== Biobjective DP Solvers ===\n\n";

  // -----------------------------------------------------------------
  // Biobjective DP tests
  // -----------------------------------------------------------------
  PointSet<> B = {{1, 10}, {2, 8}, {3, 7}, {4, 5},
                  {6, 4},  {7, 3}, {9, 2}, {10, 1}};
  std::size_t k = 3;

  auto [u_val, u_sub] = dp_max_uniformity(B, k);
  std::cout << "DP Uniformity (k=" << k << "): " << u_val << "\n";
  std::cout << "  Subset:";
  for (const auto &p : u_sub)
    std::cout << " " << p.to_string(1);
  std::cout << "\n";

  auto [c_val, c_sub] = dp_min_coverage(B, k);
  std::cout << "DP Coverage (k=" << k << "): " << c_val << "\n";
  std::cout << "  Subset:";
  for (const auto &p : c_sub)
    std::cout << " " << p.to_string(1);
  std::cout << "\n";

  auto [e_val, e_sub] = dp_min_epsilon(B, k, Sense::Maximize);
  std::cout << "DP ε-Indicator (k=" << k << "): " << e_val << "\n";
  std::cout << "  Subset:";
  for (const auto &p : e_sub)
    std::cout << " " << p.to_string(1);
  std::cout << "\n";

  // Trivial tests
  auto [c_full, c_full_sub] = dp_min_coverage(B, B.size());
  assert(std::abs(c_full) < 1e-9 && "Coverage should be 0 when k=n");
  std::cout << "\nDP Coverage (k=n): " << c_full << " ✓\n";

  auto [e_full, e_full_sub] = dp_min_epsilon(B, B.size(), Sense::Maximize);
  assert(std::abs(e_full - 1.0) < 1e-9 && "ε should be 1.0 when k=n");
  std::cout << "DP ε-Indicator (k=n): " << e_full << " ✓\n";

  std::cout << "\n=== Cross-validation: DP vs Greedy/Threshold ===\n\n";

  // Compare DP uniformity vs actual uniformity of the DP subset
  double u_check = uniformity(u_sub);
  std::cout << "DP uniformity value: " << u_val << "\n";
  std::cout << "  Actual uniformity of subset: " << u_check << "\n";
  assert(std::abs(u_val - u_check) < 1e-6 && "DP value should match actual");

  // Compare DP coverage vs actual coverage of the DP subset
  double c_check = coverage_error(B, c_sub, EuclideanDistance{});
  std::cout << "DP coverage value: " << c_val << "\n";
  std::cout << "  Actual coverage of subset: " << c_check << "\n";
  assert(std::abs(c_val - c_check) < 1e-6 && "DP value should match actual");

  // Compare DP epsilon vs actual epsilon of the DP subset
  double e_check = epsilon_indicator(B, e_sub, Sense::Maximize);
  std::cout << "DP ε-indicator value: " << e_val << "\n";
  std::cout << "  Actual ε-indicator of subset: " << e_check << "\n";
  assert(std::abs(e_val - e_check) < 1e-6 && "DP value should match actual");

  std::cout << "\n=== All tests passed ===\n";
  return 0;
}
