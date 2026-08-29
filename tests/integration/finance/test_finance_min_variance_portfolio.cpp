
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

TEST(IntegrationFinance,  Finance) {
    Interpreter interp;

    expect_contains(interp, "help", "finance_min_variance_portfolio");
    expect_contains(interp, "help", "finance_bl_implied_returns");

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);

    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    EXPECT_EQ(interp.state().matrices.at("w_ms").rows(), 2u);

    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    EXPECT_EQ(interp.state().matrices.at("pi_bl").rows(), 2u);
}

TEST(IntegrationFinance,  BesselScalar) {
    Interpreter interp;

    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-9);

    expect_ok(interp, "hh = hermite_hn(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_hn(2, 0.5), 1e-9);
}
