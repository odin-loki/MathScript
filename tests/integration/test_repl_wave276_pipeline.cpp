// MathScript Integration Tests: REPL Interpreter – Wave 276 Pipeline
//
// Wave 276 REPL smoke: topo Cech/VR + simplicial invariants, FEM 3D solve,
// quantum time-evolve psi, run-length compress, Bessel specials.

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

TEST(ReplWave276Pipeline, TopoFemCompress) {
    Interpreter interp;

    expect_contains(interp, "help", "topo_cech_complex");
    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    expect_ok(interp, "chi = topo_simplicial_euler(vr)");
    ASSERT_GT(interp.state().scalars.count("chi"), 0u);

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);
}

TEST(ReplWave276Pipeline, QuantumBessel) {
    Interpreter interp;

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);

    expect_ok(interp, "bj = bessel_j(1, 0.5)");
    expect_ok(interp, "bj1 = bessel_j1(0.5)");
    expect_ok(interp, "by0 = bessel_y0(1.0)");
    expect_ok(interp, "bz = bessel_zero_jnu(0, 1)");
}
