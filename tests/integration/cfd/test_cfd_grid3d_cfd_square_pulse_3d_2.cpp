
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

TEST(IntegrationCfd,  CfdUpwind3dTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "cfd_upwind_step_3d");

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(IntegrationCfd,  BesselHyScalar) {
    Interpreter interp;
    expect_ok(interp, "hy = bessel_hy(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hy"), ms::bessel_hy(1, 0.5), 1e-8);
}
