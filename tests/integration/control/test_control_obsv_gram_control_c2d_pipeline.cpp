
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

TEST(IntegrationControl,  ObsvGramC2d) {
    Interpreter interp;
    expect_contains(interp, "help", "control_obsv_gram(A,C)");
    expect_contains(interp, "help", "control_c2d(A,B,C,D,Ts)");

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    EXPECT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    EXPECT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(IntegrationControl,  BesselIScalar) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1), 1e-8);
}
