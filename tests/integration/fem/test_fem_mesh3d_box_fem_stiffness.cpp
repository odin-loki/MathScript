
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

TEST(IntegrationFem,  Fem3dQuantumEvolve) {
    Interpreter interp;
    expect_contains(interp, "help", "fem_solve_3d");
    expect_contains(interp, "help", "fem_lagrange_deriv");
    expect_contains(interp, "help", "quantum_time_evolve_psi");

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(IntegrationFem,  EllipFScalar) {
    Interpreter interp;
    expect_ok(interp, "ef = ellip_f(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ef"), ms::ellip_f(0.5, 0.5), 1e-8);
}
