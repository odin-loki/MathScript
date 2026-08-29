
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

TEST(IntegrationSpecial,  EfficientFrontierSharpeTail28) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_efficient_frontier");
    expect_contains(interp, "help", "finance_max_sharpe");

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(IntegrationSpecial,  BesselHScalar) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1.0), 1e-8);
}
