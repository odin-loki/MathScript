// MathScript Integration Tests: REPL Interpreter – Wave 245 Graph Bindings
//
// Wired: graph_k_core_decomposition, graph_k_core_subgraph, graph_chromatic_number.

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

TEST(ReplWave245Pipeline, GraphKCoreDecompositionBinding) {
    Interpreter interp;

    // Triangle 0-1-2 with pendant 3 on 0: cores [2,2,2,1].
    expect_ok(interp, "A = [0,1,1,1; 1,0,1,0; 1,1,0,0; 1,0,0,0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    ASSERT_GT(interp.state().matrices.count("cores"), 0u);
    const auto& cores = interp.state().matrices.at("cores");
    EXPECT_EQ(cores.rows(), 4u);
    EXPECT_EQ(cores.cols(), 1u);
    EXPECT_NEAR(cores(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(cores(1, 0), 2.0, 1e-9);
    EXPECT_NEAR(cores(2, 0), 2.0, 1e-9);
    EXPECT_NEAR(cores(3, 0), 1.0, 1e-9);

    expect_error(interp, "graph_k_core_decomposition(A, 0)");
    expect_contains(interp, "help", "graph_k_core_decomposition(A)");
}

TEST(ReplWave245Pipeline, GraphKCoreSubgraphBinding) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,1,1; 1,0,1,0; 1,1,0,0; 1,0,0,0]");
    expect_ok(interp, "S = graph_k_core_subgraph(A, 2)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    const auto& S = interp.state().matrices.at("S");
    // 2-core is the triangle on original vertices 0,1,2 → 3x3 adjacency.
    EXPECT_EQ(S.rows(), 3u);
    EXPECT_EQ(S.cols(), 3u);
    EXPECT_NEAR(S(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(S(1, 2), 1.0, 1e-9);
    EXPECT_NEAR(S(2, 0), 1.0, 1e-9);

    expect_error(interp, "graph_k_core_subgraph(A)");
    expect_contains(interp, "help", "graph_k_core_subgraph(A,k)");
}

TEST(ReplWave245Pipeline, GraphChromaticNumberBinding) {
    Interpreter interp;

    // Complete graph K3: greedy chromatic number is 3.
    expect_ok(interp, "K3 = [0,1,1; 1,0,1; 1,1,0]");
    expect_ok(interp, "chi = graph_chromatic_number(K3)");
    ASSERT_GT(interp.state().scalars.count("chi"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("chi"), 3.0, 1e-9);

    expect_contains(interp, "graph_chromatic_number(K3)", "3");
    expect_error(interp, "graph_chromatic_number(K3, 0)");
    expect_contains(interp, "help", "graph_chromatic_number(A)");
}
