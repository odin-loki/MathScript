
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

TEST(IntegrationSpecial,  FftControl) {
    Interpreter interp;

    expect_contains(interp, "help", "fft_rfft");
    expect_contains(interp, "help", "control_bode");

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 1u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(IntegrationSpecial,  SignalOdeSpecial) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "lq = legendre_q(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(2, 0.5), 1e-9);
}
