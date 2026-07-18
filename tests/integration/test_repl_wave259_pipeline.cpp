// MathScript Integration Tests: REPL Interpreter – Wave 259 Pipeline
//
// finance_efficient_frontier / finance_max_sharpe REPL bindings.

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error";
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

void expect_near_scalar(Interpreter& interp, const std::string& cmd, double expected, double tol) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    const double value = std::stod(*result);
    EXPECT_NEAR(value, expected, tol) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplWave259Pipeline, FinanceEfficientFrontier) {
    Interpreter interp;

    expect_ok(interp, "cov3 = [0.10, 0.02, 0.01; 0.02, 0.08, 0.03; 0.01, 0.03, 0.06]");
    expect_ok(interp, "mu3 = [0.08; 0.12; 0.10]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov3, mu3, 0.10)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").cols(), 1u);

    const auto& w = interp.state().matrices.at("w_ef");
    EXPECT_NEAR(w(0, 0) + w(1, 0) + w(2, 0), 1.0, 1e-8);
    expect_near_scalar(interp, "finance_portfolio_return(w_ef, mu3)", 0.10, 1e-6);

    expect_contains(interp, "help", "finance_efficient_frontier(cov,mu,target_return)");
}

TEST(ReplWave259Pipeline, FinanceMaxSharpe) {
    Interpreter interp;

    expect_ok(interp, "cov3 = [0.10, 0.02, 0.01; 0.02, 0.08, 0.03; 0.01, 0.03, 0.06]");
    expect_ok(interp, "mu3 = [0.08; 0.12; 0.10]");
    expect_ok(interp, "w_ms = finance_max_sharpe(cov3, mu3, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ms").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("w_ms").cols(), 1u);

    const auto& w = interp.state().matrices.at("w_ms");
    EXPECT_NEAR(w(0, 0) + w(1, 0) + w(2, 0), 1.0, 1e-8);

    expect_ok(interp, "w_legacy = finance_max_sharpe_portfolio(cov3, mu3, 0.02)");
    const auto& legacy = interp.state().matrices.at("w_legacy");
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(w(i, 0), legacy(i, 0), 1e-10);
    }

    expect_contains(interp, "help", "finance_max_sharpe(cov,mu,risk_free)");
}
