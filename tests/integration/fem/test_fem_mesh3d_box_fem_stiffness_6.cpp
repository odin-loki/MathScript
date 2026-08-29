
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

TEST(IntegrationFem,  FemDirichletSchrodingerFinalTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "fem_apply_dirichlet");
    expect_contains(interp, "help", "quantum_schrodinger_final");

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    EXPECT_GT(interp.state().matrices.at("sys3").rows(), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(IntegrationFem,  SphericalYnScalar) {
    Interpreter interp;
    expect_ok(interp, "sy = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::spherical_yn(1, 1.0), 1e-8);
}
