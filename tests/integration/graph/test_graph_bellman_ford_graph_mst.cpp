
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error: "
                                    << (result ? *result : "unknown");
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(IntegrationGraph,  BellmanMstKruskalTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_bellman_ford(A,source)");
    expect_contains(interp, "help", "graph_mst_kruskal(A)");

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(IntegrationGraph,  ProbExpCdfScalar) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}
