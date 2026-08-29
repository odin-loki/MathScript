
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

TEST(IntegrationGraph,  BetweennessClosenessTail30) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_betweenness(A)");
    expect_contains(interp, "help", "graph_closeness(A)");

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(IntegrationGraph,  Theta2Scalar) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}
