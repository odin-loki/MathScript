
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

TEST(IntegrationFem,  Fem1dComposable) {
    Interpreter interp;

    expect_contains(interp, "help", "fem_mesh1d(a,b,n_elements)");
    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 4)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    expect_ok(interp, "sys1 = fem_apply_dirichlet(K1, f1, [0, 4], [0, 0])");
    expect_ok(interp, "u1 = fem_solve(sys1)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 5u);

    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(IntegrationFem,  CfdComposableAdvection) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 32)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.2)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.3, 0.3)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    expect_contains(interp, "help", "cfd_run_advection_2d");
}

TEST(IntegrationFem,  QuantumIzaac) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 1; 1, 0]");
    expect_ok(interp, "Xd = quantum_dagger(X)");
    expect_ok(interp, "P = quantum_matmul_dm(X, Xd)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "bases = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("bases").rows(), 5u);

    expect_ok(interp, "izaac seed 11");
    expect_ok(interp, "rm = izaac_rand_matrix(2, 3)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 2u);
}
