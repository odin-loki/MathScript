// MathScript Integration Tests: REPL Interpreter – Wave 264 Pipeline
//
// Wave 264 REPL smoke: poly roots/fit/hermite/gcd/squarefree, numthy cornacchia,
// finance Treynor/IR, control ctrb/obsv/tf2ss/c2d, special airy_bi (+ bessel/lambert/kummer),
// info channel_capacity / blahut_arimoto.
// Regressions: Wave 263 combo_bracelets / numthy_pell_solve / poly_lagrange /
// finance_historical_var / control_kalman_predict / special_airy_ai.
// Bindings may land from sibling agents; this suite matches the intended API.

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

TEST(ReplWave264Pipeline, PolyRoots) {
    Interpreter interp;

    // p(x) = x^2 - 5x + 6 = (x-2)(x-3) → roots 2, 3 as Nx2 [re,im]
    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "r = poly_roots(p)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("r").cols(), 2u);
    expect_contains(interp, "help", "poly_roots(p)");
}

TEST(ReplWave264Pipeline, PolyFit) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pf = poly_fit(xs, ys, 2)");
    ASSERT_GT(interp.state().matrices.count("pf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pf").cols(), 1u);
    EXPECT_GE(interp.state().matrices.at("pf").rows(), 3u);
    expect_contains(interp, "help", "poly_fit(xs,ys,degree)");
}

TEST(ReplWave264Pipeline, PolyInterpHermite) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1]");
    expect_ok(interp, "ys = [0; 1]");
    expect_ok(interp, "dys = [1; 1]");
    expect_ok(interp, "ph = poly_interp_hermite(xs, ys, dys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").cols(), 1u);
    expect_contains(interp, "help", "poly_interp_hermite(xs,ys,dys)");
}

TEST(ReplWave264Pipeline, PolyGcd) {
    Interpreter interp;

    // gcd(x^2-1, x-1) = x-1 (up to scale)
    expect_ok(interp, "a = [-1; 0; 1]");
    expect_ok(interp, "b = [-1; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g").cols(), 1u);
    expect_contains(interp, "help", "poly_gcd(a,b)");
}

TEST(ReplWave264Pipeline, PolySquarefree) {
    Interpreter interp;

    // (x-1)^2 (x-2) = x^3 - 4x^2 + 5x - 2 → squarefree part (x-1)(x-2)
    expect_ok(interp, "p = [-2; 5; -4; 1]");
    expect_ok(interp, "sf = poly_squarefree(p)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sf").cols(), 1u);
    expect_contains(interp, "help", "poly_squarefree(p)");
}

TEST(ReplWave264Pipeline, NumthyCornacchia) {
    Interpreter interp;

    // cornacchia(1, 5) → (2, 1): 2^2 + 1*1^2 = 5
    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xy").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("xy").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("xy")(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("xy")(0, 1), 1.0, 1e-9);
    expect_contains(interp, "help", "numthy_cornacchia(d,p)");
}

TEST(ReplWave264Pipeline, FinanceTreynor) {
    Interpreter interp;

    // mean = 0.10, (0.10 - 0.05) / 1.2 = 0.041666...
    expect_ok(interp, "ret = [0.10; 0.12; 0.08; 0.11; 0.09]");
    expect_ok(interp, "tr = finance_treynor(ret, 0.05, 1.2)");
    ASSERT_GT(interp.state().scalars.count("tr"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("tr"), 0.05 / 1.2, 1e-9);
    expect_contains(interp, "help", "finance_treynor(returns,risk_free,beta)");
}

TEST(ReplWave264Pipeline, FinanceInformationRatio) {
    Interpreter interp;

    expect_ok(interp, "ret = [0.10; 0.12; 0.08; 0.11; 0.09]");
    expect_ok(interp, "bench = [0.08; 0.09; 0.07; 0.08; 0.07]");
    expect_ok(interp, "ir = finance_information_ratio(ret, bench)");
    ASSERT_GT(interp.state().scalars.count("ir"), 0u);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ir")));
    expect_contains(interp, "help", "finance_information_ratio(returns,benchmark)");
}

TEST(ReplWave264Pipeline, ControlCtrb) {
    Interpreter interp;

    // Double integrator: A=[[0,1],[0,0]], B=[[0],[1]] → ctrb 2x2
    expect_ok(interp, "A = [0, 1; 0, 0]");
    expect_ok(interp, "B = [0; 1]");
    expect_ok(interp, "Cmat = control_ctrb(A, B)");
    ASSERT_GT(interp.state().matrices.count("Cmat"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Cmat").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("Cmat").cols(), 2u);
    expect_contains(interp, "help", "control_ctrb(A,B)");
}

TEST(ReplWave264Pipeline, ControlObsv) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 0, 0]");
    expect_ok(interp, "C = [1, 0]");
    expect_ok(interp, "O = control_obsv(A, C)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 2u);
    expect_contains(interp, "help", "control_obsv(A,C)");
}

TEST(ReplWave264Pipeline, ControlTf2ss) {
    Interpreter interp;

    // 1/(s+1)(s+2) = 1/(s^2+3s+2) → 2-state SS; A block returned as matrix
    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [1; 3; 2]");
    expect_ok(interp, "ssA = control_tf2ss(num, den)");
    ASSERT_GT(interp.state().matrices.count("ssA"), 0u);
    EXPECT_GE(interp.state().matrices.at("ssA").rows(), 2u);
    expect_contains(interp, "help", "control_tf2ss(num,den)");
}

TEST(ReplWave264Pipeline, ControlC2d) {
    Interpreter interp;

    expect_ok(interp, "A = [-1, 0; 0, -2]");
    expect_ok(interp, "B = [1; 1]");
    expect_ok(interp, "C = [1, 0]");
    expect_ok(interp, "D = [0]");
    expect_ok(interp, "Ad = control_c2d(A, B, C, D, 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Ad").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("Ad").cols(), 2u);
    expect_contains(interp, "help", "control_c2d(");
}

TEST(ReplWave264Pipeline, SpecialAiryBi) {
    Interpreter interp;

    expect_ok(interp, "bi = special_airy_bi(0)");
    ASSERT_GT(interp.state().scalars.count("bi"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("bi"), 0.6149266274460007, 1e-6);
    expect_contains(interp, "help", "special_airy_bi(x)");
}

TEST(ReplWave264Pipeline, SpecialBesselY) {
    Interpreter interp;

    // Match existing bessel_j0 REPL naming style (unprefixed bessel_*)
    expect_ok(interp, "y0 = bessel_y(0, 1)");
    ASSERT_GT(interp.state().scalars.count("y0"), 0u);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("y0")));
    expect_contains(interp, "help", "bessel_y(");
}

TEST(ReplWave264Pipeline, SpecialLambertW) {
    Interpreter interp;

    expect_ok(interp, "w = lambert_w(0, 0)");
    ASSERT_GT(interp.state().scalars.count("w"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("w"), 0.0, 1e-12);
    expect_contains(interp, "help", "lambert_w(");
}

TEST(ReplWave264Pipeline, SpecialKummerU) {
    Interpreter interp;

    expect_ok(interp, "u = kummer_u(1, 2, 1)");
    ASSERT_GT(interp.state().scalars.count("u"), 0u);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("u")));
    expect_contains(interp, "help", "kummer_u(a,b,");
}

TEST(ReplWave264Pipeline, InfoChannelCapacity) {
    Interpreter interp;

    // Symmetric BSC-like channel; capacity ≈ channel_capacity_bsc(0.2)
    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "cap = info_channel_capacity(W)");
    ASSERT_GT(interp.state().scalars.count("cap"), 0u);
    EXPECT_GT(interp.state().scalars.at("cap"), 0.0);
    EXPECT_LT(interp.state().scalars.at("cap"), 1.0 + 1e-6);
    expect_contains(interp, "help", "info_channel_capacity(W)");
}

TEST(ReplWave264Pipeline, InfoBlahutArimoto) {
    Interpreter interp;

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "c = info_blahut_arimoto(W)");
    ASSERT_GT(interp.state().scalars.count("c"), 0u);
    EXPECT_GT(interp.state().scalars.at("c"), 0.0);
    expect_contains(interp, "help", "info_blahut_arimoto(W)");
}

// ---- Wave 263 regressions ----

TEST(ReplWave264Pipeline, ComboBraceletsRegression) {
    Interpreter interp;

    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("br").cols(), 3u);
    expect_contains(interp, "help", "combo_bracelets(n,k)");
}

TEST(ReplWave264Pipeline, NumthyPellSolveRegression) {
    Interpreter interp;

    expect_ok(interp, "pell = numthy_pell_solve(61)");
    ASSERT_GT(interp.state().matrices.count("pell"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pell").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("pell").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("pell")(0, 0), 1766319049.0, 1e-3);
    EXPECT_NEAR(interp.state().matrices.at("pell")(0, 1), 226153980.0, 1e-3);
    expect_contains(interp, "help", "numthy_pell_solve(D)");
}

TEST(ReplWave264Pipeline, PolyLagrangeRegression) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pl")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("pl")(2, 0), 1.0, 1e-8);
    expect_contains(interp, "help", "poly_lagrange(xs,ys)");
}

TEST(ReplWave264Pipeline, FinanceHistoricalVarRegression) {
    Interpreter interp;

    expect_ok(interp, "ret = [-0.20; -0.15; -0.10; -0.05; 0.0; 0.05; 0.10; 0.15; 0.20; 0.25]");
    expect_ok(interp, "hvar = finance_historical_var(ret, 0.95)");
    ASSERT_GT(interp.state().scalars.count("hvar"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("hvar"), 0.20, 1e-10);
    expect_contains(interp, "help", "finance_historical_var(returns,confidence)");
}

TEST(ReplWave264Pipeline, ControlKalmanPredictRegression) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "A = [1]");
    expect_ok(interp, "Q = [0.05]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, A, Q)");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-12);
    expect_contains(interp, "help", "control_kalman_predict(x,P,A,Q)");
}

TEST(ReplWave264Pipeline, SpecialAiryAiRegression) {
    Interpreter interp;

    expect_ok(interp, "ai = special_airy_ai(0)");
    ASSERT_GT(interp.state().scalars.count("ai"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("ai"), 0.355028053887817, 1e-6);
    expect_contains(interp, "help", "special_airy_ai(x)");
}
