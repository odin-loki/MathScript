
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

TEST(IntegrationSpecial,  QuantumMpcGbmBacktestTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_schmidt_decomposition");
    expect_contains(interp, "help", "mpc_split");
    expect_contains(interp, "help", "simulate_gbm_path");
    expect_contains(interp, "help", "run_backtest");

    expect_ok(interp, "psi = [0.5; 0.5; 0.5; 0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_EQ(interp.state().matrices.at("sh").cols(), 3u);

    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100; 101; 102; 101]");
    expect_ok(interp, "pos = [1; 1; 1; 0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(IntegrationSpecial,  BesselYScalar) {
    Interpreter interp;
    expect_ok(interp, "y = bessel_y(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ms::bessel_y(1, 1.0), 1e-8);
}
