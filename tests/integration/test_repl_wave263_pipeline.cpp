// MathScript Integration Tests: REPL Interpreter – Wave 263 Pipeline
//
// Wave 263 REPL smoke: combo bracelets/lyndon/restricted_partitions/involutions,
// numthy pell/quadratic_residues, poly lagrange/interp_newton, finance BL/merton/VaR,
// control kalman predict/update, special voigt/airy_ai.
// Regressions: Wave 262 combo_necklaces / numthy_stern_brocot.
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

TEST(ReplWave263Pipeline, ComboBracelets) {
    Interpreter interp;

    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("br").cols(), 3u);
    expect_contains(interp, "help", "combo_bracelets(n,k)");
}

TEST(ReplWave263Pipeline, ComboLyndonWords) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("lw").cols(), 3u);
    expect_contains(interp, "help", "combo_lyndon_words(n,k)");
}

TEST(ReplWave263Pipeline, ComboRestrictedPartitions) {
    Interpreter interp;

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rp").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rp").cols(), 2u);
    expect_contains(interp, "help", "combo_restricted_partitions(n,k)");
}

TEST(ReplWave263Pipeline, ComboInvolutions) {
    Interpreter interp;

    expect_ok(interp, "inv = combo_involutions(4)");
    ASSERT_GT(interp.state().scalars.count("inv"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("inv"), 10.0, 1e-12);
    expect_contains(interp, "help", "combo_involutions(n)");
}

TEST(ReplWave263Pipeline, NumthyPellSolve) {
    Interpreter interp;

    expect_ok(interp, "pell = numthy_pell_solve(61)");
    ASSERT_GT(interp.state().matrices.count("pell"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pell").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("pell").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("pell")(0, 0), 1766319049.0, 1e-3);
    EXPECT_NEAR(interp.state().matrices.at("pell")(0, 1), 226153980.0, 1e-3);
    expect_contains(interp, "help", "numthy_pell_solve(D)");
}

TEST(ReplWave263Pipeline, NumthyQuadraticResidues) {
    Interpreter interp;

    expect_ok(interp, "qr = numthy_quadratic_residues(7)");
    ASSERT_GT(interp.state().matrices.count("qr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("qr").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("qr").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("qr")(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("qr")(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("qr")(2, 0), 4.0, 1e-12);
    expect_contains(interp, "help", "numthy_quadratic_residues(p)");
}

TEST(ReplWave263Pipeline, PolyLagrange) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pl = poly_lagrange(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 1u);
    // Unique interpolant through (0,1),(1,2),(2,5) is 1 + x^2.
    EXPECT_NEAR(interp.state().matrices.at("pl")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("pl")(1, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("pl")(2, 0), 1.0, 1e-8);
    expect_contains(interp, "help", "poly_lagrange(xs,ys)");
}

TEST(ReplWave263Pipeline, PolyInterpNewton) {
    Interpreter interp;

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 2; 5]");
    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    ASSERT_GT(interp.state().matrices.count("pn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pn").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("pn").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("pn")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("pn")(1, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("pn")(2, 0), 1.0, 1e-8);
    expect_contains(interp, "help", "poly_interp_newton(xs,ys)");
}

TEST(ReplWave263Pipeline, FinanceBlImpliedReturns) {
    Interpreter interp;

    // 2-asset BL reverse-opt: Sigma=[[0.04,0.01],[0.01,0.02]], w=[0.6;0.4], delta=2.5
    // -> Pi = [0.07; 0.035]
    expect_ok(interp, "cov = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w = [0.6; 0.4]");
    expect_ok(interp, "pi = finance_bl_implied_returns(cov, w, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pi").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("pi")(0, 0), 0.07, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("pi")(1, 0), 0.035, 1e-9);
    expect_contains(interp, "help", "finance_bl_implied_returns(cov,w,delta)");
}

TEST(ReplWave263Pipeline, FinanceMertonDistanceToDefault) {
    Interpreter interp;

    expect_ok(interp, "dd = finance_merton_distance_to_default(150, 0.20, 100, 0.05, 1.0)");
    ASSERT_GT(interp.state().scalars.count("dd"), 0u);
    EXPECT_GT(interp.state().scalars.at("dd"), 1.0);
    expect_contains(interp, "help",
                    "finance_merton_distance_to_default(V,sigma_V,D,r,T)");
}

TEST(ReplWave263Pipeline, FinanceHistoricalVar) {
    Interpreter interp;

    expect_ok(interp, "ret = [-0.20; -0.15; -0.10; -0.05; 0.0; 0.05; 0.10; 0.15; 0.20; 0.25]");
    expect_ok(interp, "hvar = finance_historical_var(ret, 0.95)");
    ASSERT_GT(interp.state().scalars.count("hvar"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("hvar"), 0.20, 1e-10);
    expect_contains(interp, "help", "finance_historical_var(r[,confidence])");
}

TEST(ReplWave263Pipeline, ControlKalmanPredict) {
    Interpreter interp;

    // 1D smoke: x=0, P=1, A=1, Q=0.05 -> x_pred=0, P_pred=1.05
    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "A = [1]");
    expect_ok(interp, "Q = [0.05]");
    expect_ok(interp, "xp, Pp = control_kalman_predict(x0, P0, A, Q)");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-12);
    expect_contains(interp, "help", "control_kalman_predict(x,P,A,Q)");
}

TEST(ReplWave263Pipeline, ControlKalmanUpdate) {
    Interpreter interp;

    // 1D smoke after predict: z=2, H=1, R=0.5
    expect_ok(interp, "xp = [0]");
    expect_ok(interp, "Pp = [1.05]");
    expect_ok(interp, "H = [1]");
    expect_ok(interp, "R = [0.5]");
    expect_ok(interp, "xu, Pu = control_kalman_update(xp, Pp, 2, H, R)");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    const double S = 1.05 + 0.5;
    const double K = 1.05 / S;
    const double x_post = 0.0 + K * (2.0 - 0.0);
    const double P_post = (1.0 - K) * 1.05;
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), x_post, 1e-10);
    EXPECT_NEAR(interp.state().matrices.at("Pu")(0, 0), P_post, 1e-10);
    expect_contains(interp, "help", "control_kalman_update(x,P,z,H,R)");
}

TEST(ReplWave263Pipeline, SpecialVoigt) {
    Interpreter interp;

    expect_ok(interp, "v = special_voigt(0, 1, 0)");
    ASSERT_GT(interp.state().scalars.count("v"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("v"), 1.0 / std::sqrt(2.0 * M_PI), 1e-9);
    expect_contains(interp, "help", "special_voigt(x,sigma,gamma)");
}

TEST(ReplWave263Pipeline, SpecialAiryAi) {
    Interpreter interp;

    expect_ok(interp, "ai = special_airy_ai(0)");
    ASSERT_GT(interp.state().scalars.count("ai"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("ai"), 0.355028053887817, 1e-6);
    expect_contains(interp, "help", "special_airy_ai(x)");
}

TEST(ReplWave263Pipeline, ComboNecklacesRegression) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    ASSERT_GT(interp.state().matrices.count("neck"), 0u);
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("neck").cols(), 2u);
    expect_contains(interp, "help", "combo_necklaces(n,k)");
}

TEST(ReplWave263Pipeline, NumthySternBrocotRegression) {
    Interpreter interp;

    expect_ok(interp, "sb = numthy_stern_brocot(7)");
    ASSERT_GT(interp.state().matrices.count("sb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sb").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("sb").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("sb")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("sb")(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("sb")(6, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("sb")(6, 1), 1.0, 1e-9);
    expect_contains(interp, "help", "numthy_stern_brocot(n)");
}
