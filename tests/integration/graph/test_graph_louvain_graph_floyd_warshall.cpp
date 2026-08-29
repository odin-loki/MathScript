
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
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

TEST(IntegrationGraph,  LouvainFloydTail30) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_louvain(A)");
    expect_contains(interp, "help", "graph_floyd_warshall(A)");

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(IntegrationGraph,  LambertWScalar) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.2)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.2), 1e-8);
}
