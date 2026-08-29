
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error";
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(IntegrationFem,  Fem3dAssembly) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 1, 1, 1)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    expect_contains(interp, "help", "fem_mesh3d_box");
}

TEST(IntegrationFem,  Cfd3dPrimitives) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    expect_contains(interp, "help", "cfd_grid3d");
}

TEST(IntegrationFem,  SpecialWeierstrassZeta) {
    Interpreter interp;

    expect_ok(interp, "wz = weierstrass_zeta(0.5, 1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("wz")));
    expect_contains(interp, "help", "weierstrass_zeta");
}

TEST(IntegrationFem,  GriaGf2nField) {
    Interpreter interp;

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fld").rows(), 16u);
}

TEST(IntegrationFem,  QuantumSpectrum) {
    Interpreter interp;

    expect_ok(interp, "H = [2, 0; 0, 5]");
    expect_ok(interp, "evals = quantum_eigenspectrum(H)");
    expect_ok(interp, "gs = quantum_ground_state(H)");
    EXPECT_NEAR(interp.state().matrices.at("evals")(0, 0), 2.0, 1e-9);
}

TEST(IntegrationFem,  CrossWave236Smoke) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
}
