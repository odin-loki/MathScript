
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

TEST(IntegrationSignal,  XcorrMidpoint) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_xcorr(a,b,max_lag)");
    expect_contains(interp, "help", "signal_xcov(a,b,max_lag)");

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(IntegrationSignal,  KelvinBeiScalar) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}
