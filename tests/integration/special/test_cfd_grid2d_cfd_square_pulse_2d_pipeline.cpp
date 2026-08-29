
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

TEST(IntegrationSpecial,  CfdGridPulseTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "cfd_grid2d");
    expect_contains(interp, "help", "cfd_square_pulse_2d");

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(IntegrationSpecial,  BesselY0Scalar) {
    Interpreter interp;
    expect_ok(interp, "by0 = bessel_y0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("by0"), ms::bessel_y0(0.5), 1e-8);
}
