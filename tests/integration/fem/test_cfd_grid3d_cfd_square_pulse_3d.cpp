
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

TEST(IntegrationFem,  Cfd3dFem1dTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "cfd_run_advection_3d");
    expect_contains(interp, "help", "fem_mesh1d");

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);

    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(IntegrationFem,  BesselKScalar) {
    Interpreter interp;
    expect_ok(interp, "k = bessel_k(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("k"), ms::bessel_k(1, 1.0), 1e-8);
}
