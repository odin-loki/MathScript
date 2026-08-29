
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

TEST(IntegrationGraph,  EulerSccTail30) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_euler_circuit(A)");
    expect_contains(interp, "help", "graph_scc(A)");

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(IntegrationGraph,  AngerJScalar) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}
