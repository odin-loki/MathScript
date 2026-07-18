// MathScript Integration Tests: REPL Interpreter – Wave 240 Bindings Pipeline
//
// Wired: graph_bipartite_match, graph_biconnected_components, graph_eulerian_path,
//        finance_geo_asian_call/put, bicgstab.

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error";
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

void expect_error(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    EXPECT_FALSE(result.has_value()) << cmd;
}

} // namespace

TEST(ReplWave240Pipeline, GraphBipartiteMatchBinding) {
    Interpreter interp;

    // Complete bipartite K2,2: left={0,1}, right={2,3}; matching size 2.
    expect_ok(interp,
              "A = [0, 0, 1, 1; "
              "0, 0, 1, 1; "
              "1, 1, 0, 0; "
              "1, 1, 0, 0]");
    expect_ok(interp, "m = graph_bipartite_match(A, 2)");
    ASSERT_GT(interp.state().scalars.count("m"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("m"), 2.0, 1e-9);

    expect_error(interp, "graph_bipartite_match(A)");
    expect_contains(interp, "help", "graph_bipartite_match");
}

TEST(ReplWave240Pipeline, GraphBiconnectedComponentsBinding) {
    Interpreter interp;

    // Path 0-1-2-3: three bridge blocks.
    expect_ok(interp,
              "P = [0, 1, 0, 0; "
              "1, 0, 1, 0; "
              "0, 1, 0, 1; "
              "0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    ASSERT_GT(interp.state().matrices.count("bcc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("bcc").cols(), 3u);

    expect_contains(interp, "help", "graph_biconnected_components(A)");
}

TEST(ReplWave240Pipeline, GraphEulerianPathBinding) {
    Interpreter interp;

    // Path 0-1-2-3 has an Eulerian path (not a circuit).
    expect_ok(interp,
              "P = [0, 1, 0, 0; "
              "1, 0, 1, 0; "
              "0, 1, 0, 1; "
              "0, 0, 1, 0]");
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    ASSERT_GT(interp.state().matrices.count("ep"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ep").cols(), 1u);
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);

    expect_contains(interp, "help", "graph_eulerian_path(A)");
}

TEST(ReplWave240Pipeline, FinanceGeoAsianBindings) {
    Interpreter interp;

    expect_ok(interp, "c = finance_geo_asian_call(100, 100, 1, 0.05, 0.20, 50)");
    ASSERT_GT(interp.state().scalars.count("c"), 0u);
    EXPECT_GT(interp.state().scalars.at("c"), 0.0);
    EXPECT_LT(interp.state().scalars.at("c"), 100.0);

    expect_ok(interp, "p = finance_geo_asian_put(100, 100, 1, 0.05, 0.20, 50)");
    ASSERT_GT(interp.state().scalars.count("p"), 0u);
    EXPECT_GT(interp.state().scalars.at("p"), 0.0);
    EXPECT_LT(interp.state().scalars.at("p"), 100.0);

    expect_error(interp, "finance_geo_asian_call(100, 100, 1, 0.05, 0.20)");
    expect_contains(interp, "help", "finance_geo_asian_call");
    expect_contains(interp, "help", "finance_geo_asian_put");
}

TEST(ReplWave240Pipeline, BicgstabBinding) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("x").cols(), 1u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("x")(0, 0)));
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("x")(1, 0)));

    expect_error(interp, "bicgstab(A)");
    expect_contains(interp, "help", "bicgstab");
}
