
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

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

TEST(IntegrationPde,  PdeControl) {
    Interpreter interp;

    expect_contains(interp, "help", "pde_poisson_2d");
    expect_contains(interp, "help", "control_series");

    expect_ok(interp, "f0 = zeros(5, 5)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 100, 1e-8)");
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 5u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);

    expect_ok(interp, "B = [1,1,1,1,1; 0,0,0,0,0; 0,0,0,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "lap = pde_laplace_2d(5, 5, B)");
    EXPECT_EQ(interp.state().matrices.at("lap").rows(), 5u);

    expect_ok(interp, "num1 = [1]");
    expect_ok(interp, "den1 = [1, 1]");
    expect_ok(interp, "num2 = [2]");
    expect_ok(interp, "den2 = [1, 2]");
    expect_ok(interp, "ser = control_series(num1, den1, num2, den2)");
    EXPECT_EQ(interp.state().matrices.at("ser").rows(), 2u);

    expect_ok(interp, "ir = control_impulse_response([1], [1, 1], 5, 50)");
    expect_ok(interp, "Wc = control_ctrb_gram([-1, 0; 0, -2], [1; 1])");
    EXPECT_GT(interp.state().matrices.at("ir").rows(), 0u);
    EXPECT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(IntegrationPde,  SignalOdeSpecial) {
    Interpreter interp;

    expect_ok(interp, "h = signal_firwin(5, 0.2)");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_deconv(x, [1; 1])");
    expect_ok(interp, "xc = signal_xcorr(x, x, 2)");
    EXPECT_GT(interp.state().matrices.at("h").rows(), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 4u);
    EXPECT_GT(interp.state().matrices.at("xc").rows(), 0u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "u0 = [1; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0, 1, 0.1, 0.01, 10)");
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 3u);

    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_TRUE(interp.state().scalars.count("ek") > 0);
}
