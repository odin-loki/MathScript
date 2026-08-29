
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

TEST(IntegrationGraph,  GreedyColourKCoreTail30) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_greedy_colour(A)");
    expect_contains(interp, "help", "graph_k_core_decomposition(A)");

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(IntegrationGraph,  Theta4Scalar) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0.2, 0.1), 1e-8);
}
