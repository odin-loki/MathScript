// MathScript Integration Tests: REPL Interpreter – Wave 63–66 Bindings Pipeline

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

void expect_output_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplWave67Pipeline, Wave63_66_BindingsPipeline) {
    Interpreter interp;

    // 1. finance_bs_call + finance_bond_price scalars finite
    expect_ok(interp, "bs = finance_bs_call(100, 100, 1, 0.05, 0.2)");
    expect_ok(interp, "bond = finance_bond_price(0.05, 0.05, 10)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("bs")));
    EXPECT_GT(interp.state().scalars.at("bs"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("bond")));
    EXPECT_NEAR(interp.state().scalars.at("bond"), 100.0, 1e-6);

    // 2. graph_pagerank + graph_dijkstra_dist on small adjacency
    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_output_contains(interp, "graph_dijkstra_dist(A, 0, 2)", "3");
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
}
