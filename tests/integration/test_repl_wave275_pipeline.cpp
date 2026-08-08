// MathScript Integration Tests: REPL Interpreter – Wave 275 Pipeline
//
// Wave 275 REPL smoke: CFD mass integrals, grid-based 2D step, constant velocity,
// quantum Bell states, special functions, cellai Hebbian assign, crypto_to_hex.

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

TEST(ReplWave275Pipeline, CfdMassAndComposable2d) {
    Interpreter interp;

    expect_contains(interp, "help", "cfd_integrated_mass_1d(grid,u)");
    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 32)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.2)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    expect_ok(interp, "m1 = cfd_integrated_mass_1d(g1, u1)");
    ASSERT_GT(interp.state().scalars.count("m1"), 0u);
    EXPECT_GT(interp.state().scalars.at("m1"), 0.0);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.3, 0.3)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    expect_ok(interp, "m2 = cfd_integrated_mass_2d(g2, u3)");
    ASSERT_GT(interp.state().scalars.count("m2"), 0u);

    expect_ok(interp, "vcol = cfd_constant_velocity(8, 1.5)");
    EXPECT_EQ(interp.state().matrices.at("vcol").rows(), 8u);
}

TEST(ReplWave275Pipeline, QuantumSpecialCryptoCellai) {
    Interpreter interp;

    expect_ok(interp, "bells = quantum_bell_states()");
    EXPECT_EQ(interp.state().matrices.at("bells").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("bells").cols(), 4u);

    expect_ok(interp, "yn = spherical_yn(0, 0.5)");
    expect_ok(interp, "bh = bessel_h(1, 0.5)");
    expect_ok(interp, "hh = hermite_hn(2, 0.5)");

    expect_contains(interp, "crypto_to_hex(48656c6c6f)", "48656c6c6f");

    expect_ok(interp, "W = [0.1]");
    expect_ok(interp, "X = [1]");
    expect_ok(interp, "Y = [0.8]");
    expect_ok(interp, "W2 = cellai_hebbian_update(W, X, Y, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("W2").rows(), 1u);
}
