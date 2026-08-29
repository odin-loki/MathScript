
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

TEST(IntegrationGraph,  TopoGreedyColourTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_topological_sort");
    expect_contains(interp, "help", "graph_greedy_colour");

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);
}

TEST(IntegrationGraph,  Theta1Scalar) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}
