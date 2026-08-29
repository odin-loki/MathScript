
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

TEST(IntegrationGraph,  SccLouvainTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_scc");
    expect_contains(interp, "help", "graph_louvain");

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);

    expect_ok(interp, "B = [0,1,1,0,0,0; 1,0,1,0,0,0; 1,1,0,1,0,0; 0,0,1,0,1,1; 0,0,0,1,0,1; 0,0,0,1,1,0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);
}

TEST(IntegrationGraph,  Theta3Scalar) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0.2, 0.1), 1e-8);
}
