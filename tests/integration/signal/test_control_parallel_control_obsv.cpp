
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

TEST(IntegrationSignal,  ControlSignal) {
    Interpreter interp;

    expect_contains(interp, "help", "control_parallel");
    expect_contains(interp, "help", "signal_envelope");

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    EXPECT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "env = signal_envelope(x)");
    expect_ok(interp, "h = signal_hilbert(x)");
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "ac = signal_autocorr(x, 2)");
    EXPECT_GT(interp.state().matrices.at("ac").rows(), 0u);
}

TEST(IntegrationSignal,  OdeSpecial) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_backward_euler(\"y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "tr2 = ode_adams_bashforth2(\"y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr2").rows(), 0u);

    const double x = 0.5;
    expect_ok(interp, "lp = legendre_p(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_p(2, x), 1e-9);
}
