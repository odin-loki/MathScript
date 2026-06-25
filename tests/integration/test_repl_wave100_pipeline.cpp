// MathScript Integration Tests: REPL Interpreter – Wave 63–100 Bindings Pipeline

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplWave100Pipeline, Wave63_100_BindingsPipeline) {
    Interpreter interp;

    // --- Wave 63–66 (from integration_repl_wave67_pipeline) ---

    // 1. finance_bs_call + finance_bond_price scalars finite
    expect_ok(interp, "bs = finance_bs_call(100, 100, 1, 0.05, 0.2)");
    expect_ok(interp, "bond = finance_bond_price(0.05, 0.05, 10)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("bs")));
    EXPECT_GT(interp.state().scalars.at("bs"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("bond")));
    EXPECT_NEAR(interp.state().scalars.at("bond"), 100.0, 1e-6);

    // 2. graph_pagerank + graph_dijkstra_dist on small adjacency
    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_contains(interp, "graph_dijkstra_dist(A, 0, 2)", "3");
    expect_ok(interp, "pr = graph_pagerank(A)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(std::isfinite(interp.state().matrices.at("pr")(i, 0)));
    }

    // 3. diffgeo_gaussian_sphere + topo_euler_tetrahedron nullary scalars
    expect_ok(interp, "K = diffgeo_gaussian_sphere()");
    expect_ok(interp, "chi = topo_euler_tetrahedron()");
    EXPECT_NEAR(interp.state().scalars.at("K"), 1.0, 0.05);
    EXPECT_NEAR(interp.state().scalars.at("chi"), 1.0, 1e-9);

    // 4. tensorops_matmul 2x2 multiply correct result
    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = tensorops_matmul(M1, M2)");
    const auto& C = interp.state().matrices.at("C");
    EXPECT_NEAR(C(0, 0), 19.0, 1e-9);
    EXPECT_NEAR(C(0, 1), 22.0, 1e-9);
    EXPECT_NEAR(C(1, 0), 43.0, 1e-9);
    EXPECT_NEAR(C(1, 1), 50.0, 1e-9);

    // 5. combo_factorial + numthy_partition assignment
    expect_ok(interp, "f = combo_factorial(5)");
    expect_ok(interp, "p = numthy_partition(5)");
    EXPECT_NEAR(interp.state().scalars.at("f"), 120.0, 1e-9);
    EXPECT_NEAR(interp.state().scalars.at("p"), 7.0, 1e-9);

    // 6. ml_accuracy on simple vectors
    expect_ok(interp, "yp = [1; 0; 1; 1]");
    expect_ok(interp, "yt = [1; 0; 0; 1]");
    expect_ok(interp, "acc = ml_accuracy(yp, yt)");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("acc"), 0.75);

    // --- Wave 67–68 REPL bindings (single session) ---

    // 7. tensorops_einsum + geo_polygon_area
    expect_ok(interp, "E1 = [1, 2; 3, 4]");
    expect_ok(interp, "E2 = [5, 6; 7, 8]");
    expect_ok(interp, "Ec = tensorops_einsum(E1, E2)");
    const auto& Ec = interp.state().matrices.at("Ec");
    EXPECT_NEAR(Ec(0, 0), 19.0, 1e-9);
    EXPECT_NEAR(Ec(0, 1), 22.0, 1e-9);
    EXPECT_NEAR(Ec(1, 0), 43.0, 1e-9);
    EXPECT_NEAR(Ec(1, 1), 50.0, 1e-9);

    expect_ok(interp, "P = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "poly_area = geo_polygon_area(P)");
    EXPECT_NEAR(interp.state().scalars.at("poly_area"), 1.0, 1e-9);

    // 8. finance_irr on simple two-period cashflow
    expect_ok(interp, "cf = [-100, 110]");
    expect_ok(interp, "irr = finance_irr(cf)");
    EXPECT_NEAR(interp.state().scalars.at("irr"), 0.1, 1e-6);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("irr")));

    // 9. info_kl_divergence + info_entropy sanity
    expect_ok(interp, "p1 = [0.5; 0.5]");
    expect_ok(interp, "kl0 = info_kl_divergence(p1, p1)");
    EXPECT_NEAR(interp.state().scalars.at("kl0"), 0.0, 1e-9);
    expect_ok(interp, "kl = info_kl_divergence([0.4; 0.6], [0.5; 0.5])");
    EXPECT_GT(interp.state().scalars.at("kl"), 0.0);
    expect_ok(interp, "h1 = info_entropy(p1)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), 1.0, 1e-9);

    // 10. control_step_final (ss2tf-related)
    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [1, 1]");
    expect_ok(interp, "yfinal = control_step_final(num, den)");
    EXPECT_NEAR(interp.state().scalars.at("yfinal"), 1.0, 0.05);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("yfinal")));

    // --- Wave 69 REPL bindings (single session) ---

    // 11. control_dcgain on first-order system
    expect_ok(interp, "dc = control_dcgain([1], [1, 1])");
    EXPECT_NEAR(interp.state().scalars.at("dc"), 1.0, 1e-6);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("dc")));

    // 12. info_cross_entropy on distinct distributions
    expect_ok(interp, "ce = info_cross_entropy([0.5; 0.5], [0.25; 0.75])");
    EXPECT_GT(interp.state().scalars.at("ce"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ce")));

    // --- Wave 70 REPL bindings (single session) ---

    // 13. control_is_stable on stable first-order system
    expect_ok(interp, "stab = control_is_stable([1], [1, 1])");
    EXPECT_NEAR(interp.state().scalars.at("stab"), 1.0, 1e-6);

    // 14. finance_var on returns vector
    expect_ok(interp, "ret = [0.1; -0.05; 0.2]");
    expect_ok(interp, "var95 = finance_var(ret)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("var95")));

    // 15. info_mutual_info on independent uniform 2x2 joint PMF
    expect_ok(interp, "J = [0.25, 0.25; 0.25, 0.25]");
    expect_ok(interp, "mi = info_mutual_info(J)");
    EXPECT_NEAR(interp.state().scalars.at("mi"), 0.0, 1e-9);

    // 16. combo_nchoosek assignment
    expect_ok(interp, "c = combo_nchoosek(5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("c"), 10.0, 1e-9);

    // --- Wave 71 REPL bindings (single session) ---

    // 17. finance_cvar on returns vector
    expect_ok(interp, "cvar95 = finance_cvar(ret)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("cvar95")));

    // 18. info_js_divergence on disjoint support (see test_info InfoJSDivergence)
    expect_ok(interp, "js = info_js_divergence([1; 0], [0; 1])");
    EXPECT_NEAR(interp.state().scalars.at("js"), 1.0, 1e-9);

    // 19. quantum_von_neumann_entropy on pure-state density matrix |0><0|
    expect_ok(interp, "rho = [1, 0; 0, 0]");
    expect_ok(interp, "S = quantum_von_neumann_entropy(rho)");
    EXPECT_NEAR(interp.state().scalars.at("S"), 0.0, 1e-9);

    // --- Wave 72 REPL bindings (single session) ---

    // 20. finance_sortino on returns vector
    expect_ok(interp, "sortino = finance_sortino(ret)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("sortino")));
    EXPECT_GT(interp.state().scalars.at("sortino"), 0.0);

    // 21. info_tv_distance (see test_info InfoDivergence.TVDistance)
    expect_ok(interp, "tv = info_tv_distance([0.5; 0.5], [1; 0])");
    EXPECT_NEAR(interp.state().scalars.at("tv"), 0.5, 1e-9);

    // 22. quantum_fidelity on identical pure-state density matrices
    expect_ok(interp, "F = quantum_fidelity(rho, rho)");
    EXPECT_NEAR(interp.state().scalars.at("F"), 1.0, 1e-6);

    // --- Wave 73 REPL bindings (single session) ---

    // 23. finance_max_drawdown on equity curve (see test_finance FinanceRisk.MaxDrawdown)
    expect_ok(interp, "equity = [100; 110; 105; 120; 90; 95]");
    expect_ok(interp, "mdd = finance_max_drawdown(equity)");
    EXPECT_NEAR(interp.state().scalars.at("mdd"), 0.25, 1e-6);

    // 24. info_hellinger_dist on disjoint support (see test_info InfoDivergence.HellingerDistance)
    expect_ok(interp, "hd = info_hellinger_dist([1; 0], [0; 1])");
    EXPECT_NEAR(interp.state().scalars.at("hd"), 1.0, 1e-9);

    // 25. quantum_trace_distance on identical density matrices (rho from step 19)
    expect_ok(interp, "T = quantum_trace_distance(rho, rho)");
    EXPECT_NEAR(interp.state().scalars.at("T"), 0.0, 1e-9);

    // --- Wave 74 REPL bindings (single session) ---

    // 26. finance_kelly_fraction (see test_finance FinanceKelly.Basic)
    expect_ok(interp, "kelly = finance_kelly_fraction(0.6, 2.0)");
    EXPECT_NEAR(interp.state().scalars.at("kelly"), 0.4, 1e-9);

    // 27. info_renyi_entropy on uniform 4-PMF (see test_info InfoEntropy.Renyi)
    expect_ok(interp, "r2 = info_renyi_entropy(2.0, [0.25; 0.25; 0.25; 0.25])");
    EXPECT_NEAR(interp.state().scalars.at("r2"), 2.0, 1e-9);

    // 28. quantum_concurrence on separable 4x4 product state |00><00|
    expect_ok(interp, "rho4 = [1, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "Cq = quantum_concurrence(rho4)");
    EXPECT_NEAR(interp.state().scalars.at("Cq"), 0.0, 1e-9);

    // --- Wave 75 REPL bindings (single session) ---

    // 29. finance_compound (see test_finance FinanceTVM.Compound)
    expect_ok(interp, "fv = finance_compound(100, 0.1, 3)");
    EXPECT_NEAR(interp.state().scalars.at("fv"), 133.1, 1e-6);

    // 30. info_redundancy on skewed PMF (see test_info InfoEntropy.Redundancy)
    expect_ok(interp, "red = info_redundancy([0.9; 0.1])");
    EXPECT_GT(interp.state().scalars.at("red"), 0.0);

    // 31. quantum_entanglement_entropy on product state |00>
    expect_ok(interp, "psi = [1; 0; 0; 0]");
    expect_ok(interp, "Ee = quantum_entanglement_entropy(psi, 2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("Ee"), 0.0, 1e-9);

    // --- Wave 76 REPL bindings (single session) ---

    // 32. finance_continuous_compound (see test_finance FinanceTVM.ContinuousCompound)
    expect_ok(interp, "cc = finance_continuous_compound(100, 0.1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), 100.0 * std::exp(0.1), 1e-6);

    // 33. info_efficiency on uniform PMF (see test_info InfoEntropy.Efficiency)
    expect_ok(interp, "eff = info_efficiency([0.5; 0.5])");
    EXPECT_NEAR(interp.state().scalars.at("eff"), 1.0, 1e-9);

    // 34. quantum_expectation <0|Z|0> = 1 (see test_quantum QuantumMeasure.Expectation)
    expect_ok(interp, "ex = quantum_expectation([1; 0], [1, 0; 0, -1])");
    EXPECT_NEAR(interp.state().scalars.at("ex"), 1.0, 1e-9);

    // --- Wave 77 REPL bindings (single session) ---

    // 35. finance_pv (zero rate: -pmt*n)
    expect_ok(interp, "pv0 = finance_pv(0, 5, -10, 0)");
    EXPECT_NEAR(interp.state().scalars.at("pv0"), 50.0, 1e-9);

    // 36. info_channel_capacity_bsc (see test_info InfoChannel.BSC)
    expect_ok(interp, "cap = info_channel_capacity_bsc(0)");
    EXPECT_NEAR(interp.state().scalars.at("cap"), 1.0, 1e-9);

    // 37. quantum_expectation_dm Tr(rho Z) on |0><0| (rho from step 19)
    expect_ok(interp, "exdm = quantum_expectation_dm(rho, [1, 0; 0, -1])");
    EXPECT_NEAR(interp.state().scalars.at("exdm"), 1.0, 1e-9);

    // --- Wave 78 REPL bindings (single session) ---

    // 38. finance_fv_annuity (zero rate: -pv0 - pmt*n)
    expect_ok(interp, "fv0 = finance_fv_annuity(0, 5, -10, 0)");
    EXPECT_NEAR(interp.state().scalars.at("fv0"), 50.0, 1e-9);

    // 39. info_channel_capacity_bec (see test_info InfoChannel.BEC)
    expect_ok(interp, "capbec = info_channel_capacity_bec(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("capbec"), 0.5, 1e-9);

    // 40. quantum_inner <0|0> = 1 (see test_quantum QuantumMeasure.Inner)
    expect_ok(interp, "inn = quantum_inner([1; 0], [1; 0])");
    EXPECT_NEAR(interp.state().scalars.at("inn"), 1.0, 1e-9);

    // --- Wave 79 REPL bindings (single session) ---

    // 41. finance_pmt_annuity (zero rate: -(pv0+fv)/n)
    expect_ok(interp, "pmt = finance_pmt_annuity(0, 5, -50, 0)");
    EXPECT_NEAR(interp.state().scalars.at("pmt"), 10.0, 1e-9);

    // 42. info_shannon_hartley (see test_info InfoChannel.ShannonHartley)
    expect_ok(interp, "sh = info_shannon_hartley(1000000, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), 1e6, 1.0);

    // 43. quantum_ket_normalise on [2;0] -> unit column ket |0>
    expect_ok(interp, "psi_n = quantum_ket_normalise([2; 0])");
    ASSERT_GT(interp.state().matrices.count("psi_n"), 0u);
    const auto& psi_n = interp.state().matrices.at("psi_n");
    EXPECT_EQ(psi_n.rows(), 2u);
    EXPECT_EQ(psi_n.cols(), 1u);
    EXPECT_NEAR(psi_n(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(psi_n(1, 0), 0.0, 1e-9);
    double psi_norm_sq = 0.0;
    for (size_t i = 0; i < psi_n.rows(); ++i) {
        psi_norm_sq += psi_n(i, 0) * psi_n(i, 0);
    }
    EXPECT_NEAR(std::sqrt(psi_norm_sq), 1.0, 1e-9);

    // --- Wave 80 REPL bindings (single session) ---

    // 44. finance_binomial_call (see test_finance FinanceBinomial.PricingConsistency)
    expect_ok(interp, "bin_c = finance_binomial_call(100, 100, 1, 0.05, 0.2, 200)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("bin_c")));
    EXPECT_GT(interp.state().scalars.at("bin_c"), 0.0);

    // 45. info_differential_entropy_gaussian (see test_info InfoDifferential.Gaussian)
    expect_ok(interp, "hgauss = info_differential_entropy_gaussian(1)");
    EXPECT_NEAR(interp.state().scalars.at("hgauss"), 0.5 * std::log(2.0 * 3.14159265358979323846 *
                                                                    2.7182818284590452354),
                1e-3);

    // 46. quantum_partial_trace on 4x4 |00><00| rho4 from step 28, trace out subsystem B
    expect_ok(interp, "rhoA = quantum_partial_trace(rho4, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("rhoA"), 0u);
    const auto& rhoA = interp.state().matrices.at("rhoA");
    EXPECT_EQ(rhoA.rows(), 2u);
    EXPECT_EQ(rhoA.cols(), 2u);
    EXPECT_NEAR(rhoA(0, 0), 1.0, 1e-9);

    // --- Wave 81 REPL bindings (single session) ---

    // 47. finance_binomial_put (see test_finance FinanceBinomial.PricingConsistency)
    expect_ok(interp, "bin_p = finance_binomial_put(100, 100, 1, 0.05, 0.2, 200)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("bin_p")));
    EXPECT_GT(interp.state().scalars.at("bin_p"), 0.0);

    // 48. info_differential_entropy_uniform (see test_info / info.cpp ln|b-a|)
    expect_ok(interp, "hu = info_differential_entropy_uniform(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hu"), 0.0, 1e-9);

    // 49. finance_bs_put (see test_finance FinanceBS.CallPutParity)
    expect_ok(interp, "bs_p = finance_bs_put(100, 100, 1, 0.05, 0.2)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("bs_p")));
    EXPECT_GT(interp.state().scalars.at("bs_p"), 0.0);

    // --- Wave 82 REPL bindings (single session) ---

    // 50. finance_bs_gamma (see test_finance FinanceBS.Gamma_Positive)
    expect_ok(interp, "g = finance_bs_gamma(100, 100, 1, 0.05, 0.2)");
    EXPECT_GT(interp.state().scalars.at("g"), 0.0);

    // 51. finance_bond_duration (see test_finance FinanceBond.Duration)
    expect_ok(interp, "dur = finance_bond_duration(0, 0.05, 5)");
    EXPECT_NEAR(interp.state().scalars.at("dur"), 5.0, 1e-6);

    // 52. info_rate_distortion_gaussian (see info.cpp 0.5*log2(variance/distortion))
    expect_ok(interp, "rd = info_rate_distortion_gaussian(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("rd"), 1.0, 1e-9);

    // --- Wave 83 REPL bindings (single session) ---

    // 53. finance_bs_delta (see test_finance FinanceBS.Greeks_Delta)
    expect_ok(interp, "d = finance_bs_delta(100, 100, 1, 0.05, 0.2, 1)");
    EXPECT_GT(interp.state().scalars.at("d"), 0.0);
    EXPECT_LT(interp.state().scalars.at("d"), 1.0);

    // 54. finance_bs_vega (see test_finance FinanceBS.Vega_Positive)
    expect_ok(interp, "v = finance_bs_vega(100, 100, 1, 0.05, 0.2)");
    EXPECT_GT(interp.state().scalars.at("v"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("v")));

    // 55. finance_bond_modified_duration (zero coupon: duration/(1+y))
    expect_ok(interp, "mdur = finance_bond_modified_duration(0, 0.05, 5)");
    EXPECT_NEAR(interp.state().scalars.at("mdur"), 5.0 / 1.05, 1e-6);

    // --- Wave 84 REPL bindings (single session) ---

    // 56. finance_bs_theta (see test_finance FinanceBSGreeks.ThetaRho)
    expect_ok(interp, "th = finance_bs_theta(100, 100, 1, 0.05, 0.2, 1)");
    EXPECT_LT(interp.state().scalars.at("th"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("th")));

    // 57. finance_bs_rho (see test_finance FinanceBSGreeks.ThetaRho)
    expect_ok(interp, "rh = finance_bs_rho(100, 100, 1, 0.05, 0.2, 1)");
    EXPECT_GT(interp.state().scalars.at("rh"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("rh")));

    // 58. finance_bond_convexity (zero coupon: n*(n+1)/(1+y)^2)
    expect_ok(interp, "conv = finance_bond_convexity(0, 0.05, 5)");
    EXPECT_NEAR(interp.state().scalars.at("conv"), 30.0 / (1.05 * 1.05), 1e-6);

    // --- Wave 85 REPL bindings (single session) ---

    // 59. finance_bond_ytm (see test_finance FinanceBond.YTM)
    expect_ok(interp, "bp = finance_bond_price(0.05, 0.07, 10)");
    expect_ok(interp, "ytm = finance_bond_ytm(bp, 0.05, 10)");
    EXPECT_NEAR(interp.state().scalars.at("ytm"), 0.07, 1e-5);

    // 60. info_source_coding_rate on uniform [0.5, 0.5]
    expect_ok(interp, "p = [0.5; 0.5]");
    expect_ok(interp, "scr = info_source_coding_rate(p)");
    EXPECT_NEAR(interp.state().scalars.at("scr"), 1.0, 1e-9);

    // 61. info_tsallis_entropy q=2 on [0.5, 0.5]
    expect_ok(interp, "ts = info_tsallis_entropy(2.0, p)");
    EXPECT_NEAR(interp.state().scalars.at("ts"), 0.5, 1e-9);

    // --- Wave 86 REPL bindings (single session) ---

    // 62. finance_bs_implied_vol (see test_finance FinanceBS.ImpliedVol)
    expect_ok(interp, "mp = finance_bs_call(100, 100, 1, 0.05, 0.2)");
    expect_ok(interp, "iv = finance_bs_implied_vol(mp, 100, 100, 1, 0.05, 1)");
    EXPECT_NEAR(interp.state().scalars.at("iv"), 0.2, 1e-5);

    // 63. finance_portfolio_return (see test_finance FinancePortfolio.PortfolioReturn)
    expect_ok(interp, "w = [0.6; 0.4]");
    expect_ok(interp, "ret = [0.1; 0.05]");
    expect_ok(interp, "pr = finance_portfolio_return(w, ret)");
    EXPECT_NEAR(interp.state().scalars.at("pr"), 0.08, 1e-9);

    // 64. finance_portfolio_variance (w=[0.5;0.5], cov=[0.04,0.01;0.01,0.09])
    expect_ok(interp, "w2 = [0.5; 0.5]");
    expect_ok(interp, "cov = [0.04, 0.01; 0.01, 0.09]");
    expect_ok(interp, "pvar = finance_portfolio_variance(w2, cov)");
    EXPECT_NEAR(interp.state().scalars.at("pvar"), 0.0375, 1e-9);

    // --- Wave 87 REPL bindings (single session) ---

    // 62. info_joint_entropy on uniform 2x2 joint PMF (see test_info InfoJointEntropy)
    expect_ok(interp, "J = [0.25, 0.25; 0.25, 0.25]");
    expect_ok(interp, "hj = info_joint_entropy(J, 2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("hj"), 2.0, 1e-9);

    // 63. info_conditional_entropy on same joint (H(Y|X) = 1 bit)
    expect_ok(interp, "hc = info_conditional_entropy(J, 2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("hc"), 1.0, 1e-9);

    // 64. info_sample_entropy on [1;2;3;4;5] m=2 r=0.5 (no template matches -> 0)
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "se = info_sample_entropy(x, 2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("se"), 0.0, 1e-9);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("se")));

    // --- Wave 88 REPL bindings (single session) ---

    // 65. quantum_pauli_x 2x2 with X[0,1]=1 (see test_quantum QuantumGates.PauliX)
    expect_ok(interp, "X = quantum_pauli_x()");
    ASSERT_GT(interp.state().matrices.count("X"), 0u);
    const auto& X = interp.state().matrices.at("X");
    EXPECT_EQ(X.rows(), 2u);
    EXPECT_EQ(X.cols(), 2u);
    EXPECT_NEAR(X(0, 1), 1.0, 1e-9);

    // 66. quantum_pauli_z diagonal (see test_quantum QuantumGates.PauliZ)
    expect_ok(interp, "Z = quantum_pauli_z()");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    const auto& Z = interp.state().matrices.at("Z");
    EXPECT_EQ(Z.rows(), 2u);
    EXPECT_EQ(Z.cols(), 2u);
    EXPECT_NEAR(Z(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(Z(1, 1), -1.0, 1e-9);
    EXPECT_NEAR(Z(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(Z(1, 0), 0.0, 1e-9);

    // 67. quantum_cnot_gate 4x4 (see test_quantum QuantumGates.CNOT)
    expect_ok(interp, "CNOT = quantum_cnot_gate()");
    ASSERT_GT(interp.state().matrices.count("CNOT"), 0u);
    const auto& CNOT = interp.state().matrices.at("CNOT");
    EXPECT_EQ(CNOT.rows(), 4u);
    EXPECT_EQ(CNOT.cols(), 4u);

    // --- Wave 89 REPL bindings (single session) ---

    // 68. quantum_pauli_y 2x2 (see test_quantum QuantumCommutator.PauliXZ; REPL stores Re(Y))
    expect_ok(interp, "Y = quantum_pauli_y()");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    const auto& Y = interp.state().matrices.at("Y");
    EXPECT_EQ(Y.rows(), 2u);
    EXPECT_EQ(Y.cols(), 2u);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_NEAR(Y(i, j), 0.0, 1e-9);
        }
    }

    // 69. quantum_swap_gate 4x4 (see test_quantum swap_gate in quantum.cpp)
    expect_ok(interp, "SWAP = quantum_swap_gate()");
    ASSERT_GT(interp.state().matrices.count("SWAP"), 0u);
    const auto& SWAP = interp.state().matrices.at("SWAP");
    EXPECT_EQ(SWAP.rows(), 4u);
    EXPECT_EQ(SWAP.cols(), 4u);
    EXPECT_NEAR(SWAP(1, 2), 1.0, 1e-9);
    EXPECT_NEAR(SWAP(2, 1), 1.0, 1e-9);

    // 70. quantum_identity 2x2 diagonal ones (see test_quantum QuantumTensor.TwoQubits)
    expect_ok(interp, "I2 = quantum_identity()");
    ASSERT_GT(interp.state().matrices.count("I2"), 0u);
    const auto& I2 = interp.state().matrices.at("I2");
    EXPECT_EQ(I2.rows(), 2u);
    EXPECT_EQ(I2.cols(), 2u);
    EXPECT_NEAR(I2(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(I2(1, 1), 1.0, 1e-9);
    EXPECT_NEAR(I2(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(I2(1, 0), 0.0, 1e-9);

    // --- Wave 90 REPL bindings (single session) ---

    // 71. quantum_hadamard_gate 2x2 (see test_quantum QuantumGates.HadamardCreatesPlus)
    expect_ok(interp, "H = quantum_hadamard_gate()");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    const auto& H = interp.state().matrices.at("H");
    EXPECT_EQ(H.rows(), 2u);
    EXPECT_EQ(H.cols(), 2u);
    const double h = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(H(0, 0), h, 1e-9);
    EXPECT_NEAR(H(0, 1), h, 1e-9);
    EXPECT_NEAR(H(1, 0), h, 1e-9);
    EXPECT_NEAR(H(1, 1), -h, 1e-9);

    // 72. cplx_hyperbolic_distance(0,0,0.5,0) finite positive (see test_cplx CplxHyperbolic.Distance)
    expect_ok(interp, "hd = cplx_hyperbolic_distance(0, 0, 0.5, 0)");
    EXPECT_GT(interp.state().scalars.at("hd"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("hd")));

    // 73. info_lz_complexity on binary [0;1;0;1;1;0] finite >= 0
    expect_ok(interp, "seq = [0; 1; 0; 1; 1; 0]");
    expect_ok(interp, "lz = info_lz_complexity(seq)");
    EXPECT_GE(interp.state().scalars.at("lz"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("lz")));

    // --- Wave 91 REPL bindings (single session) ---

    // 74. quantum_pauli_plus 2x2 with P+[0,1]=1 (|0><1| ladder)
    expect_ok(interp, "Pp = quantum_pauli_plus()");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    const auto& Pp = interp.state().matrices.at("Pp");
    EXPECT_EQ(Pp.rows(), 2u);
    EXPECT_EQ(Pp.cols(), 2u);
    EXPECT_NEAR(Pp(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(Pp(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(Pp(1, 0), 0.0, 1e-9);
    EXPECT_NEAR(Pp(1, 1), 0.0, 1e-9);

    // 75. quantum_pauli_minus 2x2 with P-[1,0]=1 (|1><0| ladder)
    expect_ok(interp, "Pm = quantum_pauli_minus()");
    ASSERT_GT(interp.state().matrices.count("Pm"), 0u);
    const auto& Pm = interp.state().matrices.at("Pm");
    EXPECT_EQ(Pm.rows(), 2u);
    EXPECT_EQ(Pm.cols(), 2u);
    EXPECT_NEAR(Pm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(Pm(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(Pm(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(Pm(1, 1), 0.0, 1e-9);

    // 76. quantum_toffoli_gate 8x8 (flips |110> and |111>)
    expect_ok(interp, "T = quantum_toffoli_gate()");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    const auto& T = interp.state().matrices.at("T");
    EXPECT_EQ(T.rows(), 8u);
    EXPECT_EQ(T.cols(), 8u);
    EXPECT_NEAR(T(6, 6), 0.0, 1e-9);
    EXPECT_NEAR(T(6, 7), 1.0, 1e-9);
    EXPECT_NEAR(T(7, 6), 1.0, 1e-9);
    EXPECT_NEAR(T(7, 7), 0.0, 1e-9);
    for (size_t i = 0; i < 8u; ++i) {
        if (i != 6u && i != 7u) {
            EXPECT_NEAR(T(i, i), 1.0, 1e-9);
        }
    }

    // --- Wave 92 REPL bindings (single session) ---

    // 77. quantum_rotation_z(pi/2) 2x2 Z-rotation gate (diagonal cos(pi/4))
    expect_ok(interp, "pi = 3.14159265358979323846");
    expect_ok(interp, "Rz = quantum_rotation_z(pi/2)");
    ASSERT_GT(interp.state().matrices.count("Rz"), 0u);
    const auto& Rz = interp.state().matrices.at("Rz");
    EXPECT_EQ(Rz.rows(), 2u);
    EXPECT_EQ(Rz.cols(), 2u);
    const double rz_h = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(Rz(0, 0), rz_h, 1e-9);
    EXPECT_NEAR(Rz(1, 1), rz_h, 1e-9);
    EXPECT_NEAR(Rz(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(Rz(1, 0), 0.0, 1e-9);

    // 78. quantum_rotation_x(pi/2) 2x2 X-rotation gate (diagonal cos(pi/4))
    expect_ok(interp, "Rx = quantum_rotation_x(pi/2)");
    ASSERT_GT(interp.state().matrices.count("Rx"), 0u);
    const auto& Rx = interp.state().matrices.at("Rx");
    EXPECT_EQ(Rx.rows(), 2u);
    EXPECT_EQ(Rx.cols(), 2u);
    EXPECT_NEAR(Rx(0, 0), rz_h, 1e-9);
    EXPECT_NEAR(Rx(1, 1), rz_h, 1e-9);

    // 79. quantum_rotation_y(pi/2) 2x2 Y-rotation gate (off-diagonal sin(pi/4))
    expect_ok(interp, "Ry = quantum_rotation_y(pi/2)");
    ASSERT_GT(interp.state().matrices.count("Ry"), 0u);
    const auto& Ry = interp.state().matrices.at("Ry");
    EXPECT_EQ(Ry.rows(), 2u);
    EXPECT_EQ(Ry.cols(), 2u);
    EXPECT_NEAR(Ry(0, 0), rz_h, 1e-9);
    EXPECT_NEAR(Ry(1, 1), rz_h, 1e-9);
    EXPECT_NEAR(Ry(0, 1), -rz_h, 1e-9);
    EXPECT_NEAR(Ry(1, 0), rz_h, 1e-9);

    // --- Wave 93 REPL bindings (single session) ---

    // 80. quantum_phase_gate(1.57) 2x2 phase gate — all entries finite
    expect_ok(interp, "P = quantum_phase_gate(1.57)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    const auto& P = interp.state().matrices.at("P");
    EXPECT_EQ(P.rows(), 2u);
    EXPECT_EQ(P.cols(), 2u);
    for (size_t i = 0; i < 2u; ++i) {
        for (size_t j = 0; j < 2u; ++j) {
            EXPECT_TRUE(std::isfinite(P(i, j)));
        }
    }
    EXPECT_NEAR(P(0, 0), 1.0, 1e-9);

    // 81. quantum_qft_gate(2) 4x4 QFT (see test_quantum QuantumQFT.Unitarity)
    expect_ok(interp, "Q = quantum_qft_gate(2)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    const auto& Q = interp.state().matrices.at("Q");
    EXPECT_EQ(Q.rows(), 4u);
    EXPECT_EQ(Q.cols(), 4u);

    // 82. cplx_poisson_kernel(0,0,0.5) finite positive (see cplx.cpp Poisson kernel)
    expect_ok(interp, "pk = cplx_poisson_kernel(0, 0, 0.5)");
    EXPECT_GT(interp.state().scalars.at("pk"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("pk")));
    EXPECT_NEAR(interp.state().scalars.at("pk"), 3.0, 1e-9);

    // --- Wave 94 REPL bindings (single session) ---

    // 83. tensorops_inner on 3x1 vectors -> 32 (1*4+2*5+3*6)
    expect_ok(interp, "V1 = [1; 2; 3]");
    expect_ok(interp, "V2 = [4; 5; 6]");
    expect_ok(interp, "ti = tensorops_inner(V1, V2)");
    EXPECT_NEAR(interp.state().scalars.at("ti"), 32.0, 1e-9);

    // 84. geo_dist3d(0,0,0,3,4,12) -> 13 (3-4-12 right triangle)
    expect_ok(interp, "d3 = geo_dist3d(0, 0, 0, 3, 4, 12)");
    EXPECT_NEAR(interp.state().scalars.at("d3"), 13.0, 1e-9);

    // 85. numthy_isprime(17) -> 1
    expect_ok(interp, "ip = numthy_isprime(17)");
    EXPECT_NEAR(interp.state().scalars.at("ip"), 1.0, 1e-9);

    // --- Wave 95 REPL bindings (single session) ---

    // 86. combo_stirling2(4,2) -> 7 (see test_combo ComboStirling2)
    expect_ok(interp, "s2 = combo_stirling2(4, 2)");
    EXPECT_NEAR(interp.state().scalars.at("s2"), 7.0, 1e-9);

    // 87. numthy_euler_phi(12) -> 4 (see test_numthy NumthyEulerPhi)
    expect_ok(interp, "phi = numthy_euler_phi(12)");
    EXPECT_NEAR(interp.state().scalars.at("phi"), 4.0, 1e-9);

    // 88. numthy_mobius(6) -> 1 (see test_numthy NumthyMobius)
    expect_ok(interp, "mu = numthy_mobius(6)");
    EXPECT_NEAR(interp.state().scalars.at("mu"), 1.0, 1e-9);

    // --- Wave 96 REPL bindings (single session) ---

    // 89. combo_catalan(4) -> 14 (see test_combo ComboCatalan)
    expect_ok(interp, "cat = combo_catalan(4)");
    EXPECT_NEAR(interp.state().scalars.at("cat"), 14.0, 1e-9);

    // 90. combo_bell(4) -> 15 (see test_combo ComboBell)
    expect_ok(interp, "bell = combo_bell(4)");
    EXPECT_NEAR(interp.state().scalars.at("bell"), 15.0, 1e-9);

    // 91. numthy_num_divisors(12) -> 6 (see test_numthy NumthyDivisors / tau)
    expect_ok(interp, "tau = numthy_num_divisors(12)");
    EXPECT_NEAR(interp.state().scalars.at("tau"), 6.0, 1e-9);

    // --- Wave 97 REPL bindings (single session) ---

    // 92. combo_stirling1(4,2) -> 11 (see test_combo ComboSpecial.Stirling)
    expect_ok(interp, "s1 = combo_stirling1(4, 2)");
    EXPECT_NEAR(interp.state().scalars.at("s1"), 11.0, 1e-9);

    // 93. numthy_lcm(4,6) -> 12 (see test_numthy NumthyArith)
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-9);

    // 94. numthy_mod_pow(2,10,1000) -> 24 (see test_numthy NumthyArith.ModPow)
    expect_ok(interp, "mp = numthy_mod_pow(2, 10, 1000)");
    EXPECT_NEAR(interp.state().scalars.at("mp"), 24.0, 1e-9);

    // --- Wave 98 REPL bindings (single session) ---

    // 95. combo_motzkin(4) -> 9 (see test_combo ComboSpecial.MotzkinNum)
    expect_ok(interp, "motz = combo_motzkin(4)");
    EXPECT_NEAR(interp.state().scalars.at("motz"), 9.0, 1e-9);

    // 96. combo_permutations(5,2) -> 20 (see test_combo ComboCounting.Permutations)
    expect_ok(interp, "perm = combo_permutations(5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("perm"), 20.0, 1e-9);

    // 97. numthy_sum_divisors(12) -> 28 (see test_numthy NumthyDivisors.Basic)
    expect_ok(interp, "sigma = numthy_sum_divisors(12)");
    EXPECT_NEAR(interp.state().scalars.at("sigma"), 28.0, 1e-9);

    // --- Wave 99 REPL bindings (single session) ---

    // 98. numthy_nextprime(10) -> 11 (see test_numthy NumthyPrimality.NextPrevPrime)
    expect_ok(interp, "np = numthy_nextprime(10)");
    EXPECT_NEAR(interp.state().scalars.at("np"), 11.0, 1e-9);

    // 99. numthy_liouville(12) -> -1 (see test_numthy NumthyMult.Liouville)
    expect_ok(interp, "lam = numthy_liouville(12)");
    EXPECT_NEAR(interp.state().scalars.at("lam"), -1.0, 1e-9);

    // 100. combo_subfactorial(4) -> 9 (see test_combo ComboFactorials.Subfactorial)
    expect_ok(interp, "subf = combo_subfactorial(4)");
    EXPECT_NEAR(interp.state().scalars.at("subf"), 9.0, 1e-9);

    // --- Wave 100 REPL bindings (single session) ---

    // 101. numthy_prime_pi(100) -> 25 (see test_numthy NumthySieve.PrimePi)
    expect_ok(interp, "pi = numthy_prime_pi(100)");
    EXPECT_NEAR(interp.state().scalars.at("pi"), 25.0, 1e-9);

    // 102. numthy_legendre_symbol(2,7) -> 1 (see test_numthy NumthyModular.LegendreSymbol)
    expect_ok(interp, "ls = numthy_legendre_symbol(2, 7)");
    EXPECT_NEAR(interp.state().scalars.at("ls"), 1.0, 1e-9);

    // 103. combo_combinations_with_rep(3,2) -> 6 (see test_combo ComboCounting.CombinationsWithRep)
    expect_ok(interp, "cwr = combo_combinations_with_rep(3, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cwr"), 6.0, 1e-9);
}
