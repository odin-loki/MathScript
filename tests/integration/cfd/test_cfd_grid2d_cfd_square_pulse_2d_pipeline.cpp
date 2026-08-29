
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

TEST(IntegrationCfd,  CfdUpwind2dHebbian) {
    Interpreter interp;
    expect_contains(interp, "help", "cfd_upwind_step_2d");
    expect_contains(interp, "help", "cellai_hebbian_update");

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(IntegrationCfd,  Theta4Scalar) {
    Interpreter interp;
    expect_ok(interp, "t4 = theta4(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("t4"), ms::theta4(0.2, 0.3), 1e-8);
}
