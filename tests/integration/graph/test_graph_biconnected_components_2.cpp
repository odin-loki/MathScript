
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/prob/prob.hpp"

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

TEST(IntegrationGraph,  BiconnectedEulerianTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_biconnected_components");
    expect_contains(interp, "help", "graph_eulerian_path");

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(IntegrationGraph,  ProbChi2PpfScalar) {
    Interpreter interp;
    expect_ok(interp, "q = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("q"), ms::chi2_ppf(0.5, 2), 1e-8);
}
