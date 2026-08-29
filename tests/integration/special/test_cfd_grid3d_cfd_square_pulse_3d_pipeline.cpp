
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

TEST(IntegrationSpecial,  Cfd3dGridPulseTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "cfd_grid3d");
    expect_contains(interp, "help", "cfd_square_pulse_3d");

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);
}

TEST(IntegrationSpecial,  BesselHScalar) {
    Interpreter interp;
    expect_ok(interp, "h = bessel_h(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("h"), ms::bessel_h(1, 1.0), 1e-8);
}
