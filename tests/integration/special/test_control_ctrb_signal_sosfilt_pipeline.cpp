
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

TEST(IntegrationSpecial,  CtrbSosfilt) {
    Interpreter interp;
    expect_contains(interp, "help", "control_ctrb");
    expect_contains(interp, "help", "signal_sosfilt");

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    EXPECT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(IntegrationSpecial,  ChebyshevTScalar) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}
