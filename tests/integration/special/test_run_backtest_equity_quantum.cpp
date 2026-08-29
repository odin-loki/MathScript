
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

TEST(IntegrationSpecial,  BacktestEquitySchrodinger) {
    Interpreter interp;
    expect_contains(interp, "help", "run_backtest_equity");
    expect_contains(interp, "help", "quantum_schrodinger");

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);
}

TEST(IntegrationSpecial,  StruveHScalar) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1.0), 1e-8);
}
