// MathScript Integration Tests: REPL Interpreter – Wave 261 Pipeline
//
// eig/ldl, cg/gmres/jacobi, fftfreq/ifftshift, combo_eulerian/combo_gray_code,
// numthy_factor_exp/numthy_is_carmichael, stats_weighted_variance REPL bindings.
// Smoke: Wave 260 APIs on main (minres, fft_goertzel, stats_vif,
// control_step_response). Deterministic; Wave 261 features merge from sibling
// branches before this pipeline lands last.

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

TEST(ReplWave261Pipeline, Minres) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "b = [1; 1; 1]");
    expect_ok(interp, "x = minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("x").cols(), 1u);
    expect_contains(interp, "help", "minres(A");
}

TEST(ReplWave261Pipeline, FftGoertzel) {
    Interpreter interp;

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_EQ(interp.state().matrices.at("G").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("G").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 4.0, 1e-9);
    expect_contains(interp, "help", "fft_goertzel(x,f,fs)");
}

TEST(ReplWave261Pipeline, StatsVif) {
    Interpreter interp;

    expect_ok(interp, "Xo = [1, 1; 1, -1; 1, 1; 1, -1; 1, 1; 1, -1]");
    expect_ok(interp, "v0 = stats_vif(Xo, 0)");
    ASSERT_GT(interp.state().scalars.count("v0"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("v0"), 1.0, 0.2);
    expect_contains(interp, "help", "stats_vif(X,j)");
}

TEST(ReplWave261Pipeline, ControlStepResponse) {
    Interpreter interp;

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 100)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_EQ(step.rows(), 100u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.08);
    expect_contains(interp, "help", "control_step_response(num,den[,t_end[,n_pts]])");
}

TEST(ReplWave261Pipeline, Eig) {
    Interpreter interp;

    expect_ok(interp, "Ag = [4, 1; 2, 3]");
    expect_ok(interp, "Dg, Vg = eig(Ag)");
    ASSERT_GT(interp.state().matrices.count("Dg"), 0u);
    ASSERT_GT(interp.state().matrices.count("Vg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dg").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("Vg").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("Vg").cols(), 2u);
    const double trace =
        interp.state().matrices.at("Dg")(0, 0) + interp.state().matrices.at("Dg")(1, 0);
    EXPECT_NEAR(trace, 7.0, 1e-4);
    expect_contains(interp, "help", "eig(A)");
}

TEST(ReplWave261Pipeline, Ldl) {
    Interpreter interp;

    expect_ok(interp, "Aldl = [2, 1; 1, -1]");
    expect_ok(interp, "Lldl, Dldl = ldl(Aldl)");
    ASSERT_GT(interp.state().matrices.count("Lldl"), 0u);
    ASSERT_GT(interp.state().matrices.count("Dldl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Lldl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("Lldl").cols(), 2u);
    EXPECT_EQ(interp.state().matrices.at("Dldl").rows(), 2u);
    expect_contains(interp, "help", "ldl(A)");
}

TEST(ReplWave261Pipeline, Cg) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("xc")(1, 0), 3.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("xc")(2, 0), 4.0, 1e-6);
    expect_contains(interp, "help", "cg(A");
}

TEST(ReplWave261Pipeline, Gmres) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    ASSERT_GT(interp.state().matrices.count("xg"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("xg")(1, 0), 3.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("xg")(2, 0), 4.0, 1e-6);
    expect_contains(interp, "help", "gmres(A");
}

TEST(ReplWave261Pipeline, Jacobi) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("xj")(1, 0), 3.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("xj")(2, 0), 4.0, 1e-6);
    expect_contains(interp, "help", "jacobi(A");
}

TEST(ReplWave261Pipeline, Fftfreq) {
    Interpreter interp;

    expect_ok(interp, "f8 = fftfreq(8, 1)");
    ASSERT_GT(interp.state().matrices.count("f8"), 0u);
    EXPECT_EQ(interp.state().matrices.at("f8").rows(), 8u);
    EXPECT_EQ(interp.state().matrices.at("f8").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("f8")(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("f8")(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("f8")(4, 0), -0.5, 1e-12);
    expect_contains(interp, "help", "fftfreq(n[,d])");
}

TEST(ReplWave261Pipeline, Ifftshift) {
    Interpreter interp;

    expect_ok(interp, "S = [0, 0; 1, 0; 2, 0; 3, 0]");
    expect_ok(interp, "Sh = fftshift(S)");
    expect_ok(interp, "Sr = ifftshift(Sh)");
    ASSERT_GT(interp.state().matrices.count("Sr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Sr").rows(), 4u);
    for (size_t i = 0; i < 4u; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("Sr")(i, 0), static_cast<double>(i), 1e-9);
        EXPECT_NEAR(interp.state().matrices.at("Sr")(i, 1), 0.0, 1e-9);
    }
    expect_contains(interp, "help", "ifftshift(S)");
}

TEST(ReplWave261Pipeline, ComboEulerian) {
    Interpreter interp;

    expect_ok(interp, "eu = combo_eulerian(4, 1)");
    ASSERT_GT(interp.state().scalars.count("eu"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("eu"), 11.0, 1e-9);
    expect_contains(interp, "help", "combo_eulerian(n,k)");
}

TEST(ReplWave261Pipeline, ComboGrayCode) {
    Interpreter interp;

    expect_ok(interp, "gc = combo_gray_code(2)");
    ASSERT_GT(interp.state().matrices.count("gc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gc").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("gc").cols(), 2u);
    expect_contains(interp, "help", "combo_gray_code(n)");
}

TEST(ReplWave261Pipeline, NumthyFactorExp) {
    Interpreter interp;

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    ASSERT_GT(interp.state().matrices.count("fe"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("fe").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("fe")(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("fe")(0, 1), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("fe")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("fe")(1, 1), 1.0, 1e-9);
    expect_contains(interp, "help", "numthy_factor_exp(n)");
}

TEST(ReplWave261Pipeline, NumthyIsCarmichael) {
    Interpreter interp;

    expect_ok(interp, "c561 = numthy_is_carmichael(561)");
    ASSERT_GT(interp.state().scalars.count("c561"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("c561"), 1.0, 1e-9);

    expect_ok(interp, "c97 = numthy_is_carmichael(97)");
    ASSERT_GT(interp.state().scalars.count("c97"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("c97"), 0.0, 1e-9);
    expect_contains(interp, "help", "numthy_is_carmichael(n)");
}

TEST(ReplWave261Pipeline, StatsWeightedVariance) {
    Interpreter interp;

    expect_ok(interp, "xv = [2; 4; 6; 8; 10]");
    expect_ok(interp, "wv = [1; 1; 1; 1; 1]");
    expect_ok(interp, "vw = stats_weighted_variance(xv, wv)");
    expect_ok(interp, "sv = stats_var(xv)");
    ASSERT_GT(interp.state().scalars.count("vw"), 0u);
    ASSERT_GT(interp.state().scalars.count("sv"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("vw"), interp.state().scalars.at("sv"), 1e-9);
    expect_contains(interp, "help", "stats_weighted_variance(x,w)");
}
