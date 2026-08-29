
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

TEST(IntegrationFem,  FemStiffnessLoadTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "fem_stiffness_2d");
    expect_contains(interp, "help", "fem_load_2d");

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(IntegrationFem,  BesselJ0Scalar) {
    Interpreter interp;
    expect_ok(interp, "bj0 = bessel_j0(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("bj0"), ms::bessel_j0(0.5), 1e-8);
}
