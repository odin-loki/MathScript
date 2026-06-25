// MathScript Integration Tests: REPL Interpreter – Wave 63–75 Bindings Pipeline

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

TEST(ReplWave75Pipeline, Wave63_75_BindingsPipeline) {
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
}
