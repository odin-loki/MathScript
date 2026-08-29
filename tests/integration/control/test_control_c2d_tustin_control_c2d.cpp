
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

TEST(IntegrationControl,  ControlC2dD2cTustinEuler) {
    Interpreter interp;
    expect_contains(interp, "help", "control_c2d_tustin(A,B,C,D,Ts)");
    expect_contains(interp, "help", "control_c2d_euler(A,B,C,D,Ts)");

    const double Ts = 0.1;
    const double A = -2.0;
    const double B = 3.0;
    const double tustin_denom = 1.0 - A * Ts * 0.5;
    const double Ad_tustin = (1.0 + A * Ts * 0.5) / tustin_denom;
    const double Bd_tustin = B * Ts / tustin_denom;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    EXPECT_GT(interp.state().matrices.at("Ad").rows(), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), Ad_tustin, 1e-9);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    EXPECT_GT(interp.state().matrices.at("Ad_e").rows(), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-12);

    expect_ok(interp, "Bd = [" + std::to_string(Bd_tustin) + "]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-6);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    EXPECT_NEAR(interp.state().matrices.at("Ac_e")(0, 0), -2.0, 1e-6);
}

TEST(IntegrationControl,  KelvinBerScalar) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1), 1e-8);
}
