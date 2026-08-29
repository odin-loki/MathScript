
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

TEST(IntegrationControl,  Ss2tfD2cTail30) {
    Interpreter interp;
    expect_contains(interp, "help", "control_ss2tf(SS)");
    expect_contains(interp, "help", "control_d2c(A,B,C,D,Ts)");

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_GT(tf.rows(), 0u);
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(IntegrationControl,  HermiteHfScalar) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}
