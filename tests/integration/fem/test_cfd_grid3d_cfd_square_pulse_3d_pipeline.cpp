
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

TEST(IntegrationFem,  CfdIntegratedMass3d) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "mass3 = cfd_integrated_mass_3d(g3, u0)");
    EXPECT_GT(interp.state().scalars.at("mass3"), 0.0);
    expect_contains(interp, "help", "cfd_integrated_mass_3d");
}

TEST(IntegrationFem,  QuantumAndSpecial) {
    Interpreter interp;

    expect_ok(interp, "ed = ellip_d(0.5)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ed")));

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("sch"), 0u);

    expect_ok(interp, "ac = quantum_anticommutator([1,0;0,1], [0,1;1,0])");
    ASSERT_EQ(interp.state().matrices.at("ac").rows(), 2u);
}

TEST(IntegrationFem,  IzaacFrameworks) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 7");
    expect_ok(interp, "sh = mpc_split(99, 4, 2)");
    expect_ok(interp, "sec = mpc_reconstruct(sh)");
    EXPECT_NEAR(interp.state().scalars.at("sec"), 99.0, 1e-9);
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
}

TEST(IntegrationFem,  AxiomGriaFem3d) {
    Interpreter interp;

    expect_ok(interp, "fit = axiom_gria_fitness(\"x0\", [1, 2; 3, 4])");
    EXPECT_GE(interp.state().scalars.at("fit"), 0.0);

    expect_ok(interp, "a = gria_dispatch_hint_register(\"pipe_op\", 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("a"), 0.5, 1e-12);

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 1, 1, 1)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    expect_ok(interp, "u3 = fem_solve(sys3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
}
