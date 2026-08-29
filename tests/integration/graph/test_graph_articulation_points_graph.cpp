
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

TEST(IntegrationGraph,  ArticulationBridgesTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_articulation_points(A)");
    expect_contains(interp, "help", "graph_bridges(A)");

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(IntegrationGraph,  JordanTotientScalar) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}
