
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

TEST(IntegrationSpecial,  MertonBlDefaultOmegaTail28) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_merton_implied_asset_params");
    expect_contains(interp, "help", "finance_bl_posterior_returns_default_omega");

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(IntegrationSpecial,  BesselYScalar) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1.0), 1e-8);
}
