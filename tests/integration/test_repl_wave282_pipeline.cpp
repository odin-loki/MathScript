// MathScript Integration Tests: REPL Interpreter – Wave 282 Pipeline
//
// Wave 282 REPL smoke: PDE/FEM/control/signal/ODE/sparse tail11 migrations,
// quantum_hadamard, wavelet_compress_vec, scalar debye/clausen/eta_dirichlet.

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

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

TEST(ReplWave282Pipeline, PdeFemControl) {
    Interpreter interp;

    expect_contains(interp, "help", "pde_heat_1d(");
    expect_contains(interp, "help", "control_ctrb(");

    expect_ok(interp, "x0 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "h1 = pde_heat_1d(x0, 0.1, 0.1, 0.01, 10)");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x0, 0.1, 0.1, 0.01, 10)");
    EXPECT_EQ(interp.state().matrices.at("h1").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);

    expect_ok(interp, "u0 = zeros(7, 7)");
    expect_ok(interp, "v0 = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0, v0, 1.0, 0.1, 0.1, 0.02, 8)");
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);

    expect_ok(interp, "u1 = fem_poisson1d(8)");
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);

    expect_ok(interp, "x0k = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "A = [1]");
    expect_ok(interp, "Q = [0.05]");
    expect_ok(interp, "xp = control_kalman_predict(x0k, P0, A, Q)");
    expect_ok(interp, "xu = control_kalman_update(xp, P0, [2], [1], [0.5])");
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-12);

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    EXPECT_GT(interp.state().matrices.at("Co").rows(), 0u);
}

TEST(ReplWave282Pipeline, SignalSparseOdeQuantum) {
    Interpreter interp;

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    EXPECT_GT(interp.state().matrices.at("WC").rows(), 0u);
}

TEST(ReplWave282Pipeline, SpecialScalars) {
    Interpreter interp;

    expect_ok(interp, "db = debye(1, 0.5)");
    expect_ok(interp, "cl = clausen(0.3)");
    expect_ok(interp, "et = eta_dirichlet(2.0)");
    EXPECT_TRUE(interp.state().scalars.count("db") > 0);
    EXPECT_TRUE(interp.state().scalars.count("cl") > 0);
    EXPECT_TRUE(interp.state().scalars.count("et") > 0);
}
