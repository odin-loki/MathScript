
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

TEST(IntegrationFinance,  BlImpliedPostTail28) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bl_implied_returns");
    expect_contains(interp, "help", "finance_bl_posterior_returns");

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(IntegrationFinance,  SphericalInScalar) {
    Interpreter interp;
    expect_ok(interp, "si = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("si"), ms::spherical_in(0, 1.0), 1e-8);
}
