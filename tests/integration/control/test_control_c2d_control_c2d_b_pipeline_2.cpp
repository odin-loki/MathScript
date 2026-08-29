
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

TEST(IntegrationControl,  D2cTfTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "control_d2c");
    expect_contains(interp, "help", "control_c2d_tf");
    expect_contains(interp, "help", "control_d2c_tf");

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
    expect_ok(interp, "C = control_d2c_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
}

TEST(IntegrationControl,  EllipFScalar) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}
