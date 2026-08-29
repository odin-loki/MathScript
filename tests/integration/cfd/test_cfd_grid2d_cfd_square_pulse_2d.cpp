
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

TEST(IntegrationCfd,  CfdUpwind2dTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "cfd_upwind_step_2d");

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);
}

TEST(IntegrationCfd,  ChebyshevVScalar) {
    Interpreter interp;
    expect_ok(interp, "v = chebyshev_v(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("v"), ms::chebyshev_v(1, 0.5), 1e-8);
}
