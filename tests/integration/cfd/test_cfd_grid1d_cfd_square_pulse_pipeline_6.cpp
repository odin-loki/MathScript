
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

TEST(IntegrationCfd,  CfdUpwind1dConstVelTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "cfd_upwind_step_1d");
    expect_contains(interp, "help", "cfd_constant_velocity");

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(IntegrationCfd,  HermiteHeScalar) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}
