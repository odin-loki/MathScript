
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

TEST(IntegrationOde,  TrapezoidalPagerankTail30) {
    Interpreter interp;
    expect_contains(interp, "help", "ode_trapezoidal(\"formula\",t0,y0,t_end,steps)");
    expect_contains(interp, "help", "graph_pagerank(A)");

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(IntegrationOde,  Theta1Scalar) {
    Interpreter interp;
    expect_ok(interp, "th = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta1(0.2, 0.1), 1e-8);
}
