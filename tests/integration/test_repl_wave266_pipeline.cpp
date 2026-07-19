// MathScript Integration Tests: REPL Interpreter – Wave 266 Pipeline
//
// Wave 266 REPL smoke: poly cheb_expand, control c2d_tustin, special erfi,
// prob chi2_ppf, graph min_arborescence, image imfilter/sobel_x, stats ks_norm.
// Regressions: Wave 265 poly_shift, control_series, bessel_k, info_normalized_entropy.
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

TEST(ReplWave266Pipeline, PolyChebExpand) {
    Interpreter interp;

    // x^3 - 2x + 1; Chebyshev expansion should round-trip via poly_cheb_eval
    expect_ok(interp, "p = [1; -2; 0; 1]");
    expect_ok(interp, "c = poly_cheb_expand(p, 3)");
    ASSERT_GT(interp.state().matrices.count("c"), 0u);
    EXPECT_GE(interp.state().matrices.at("c").rows(), 1u);
    expect_ok(interp, "v = poly_cheb_eval(c, 0)");
    ASSERT_GT(interp.state().scalars.count("v"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("v"), 1.0, 1e-6);
    expect_contains(interp, "help", "poly_cheb_expand(");
}

TEST(ReplWave266Pipeline, ControlC2dTustin) {
    Interpreter interp;

    expect_ok(interp, "A = [-1, 0; 0, -2]");
    expect_ok(interp, "B = [1; 1]");
    expect_ok(interp, "C = [1, 0]");
    expect_ok(interp, "D = [0]");
    expect_ok(interp, "Ad = control_c2d_tustin(A, B, C, D, 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Ad").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("Ad").cols(), 2u);
    expect_contains(interp, "help", "control_c2d_tustin(");
}

TEST(ReplWave266Pipeline, SpecialErfi) {
    Interpreter interp;

    expect_ok(interp, "e0 = erfi(0)");
    ASSERT_GT(interp.state().scalars.count("e0"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("e0"), 0.0, 1e-12);
    expect_ok(interp, "e1 = erfi(1)");
    ASSERT_GT(interp.state().scalars.count("e1"), 0u);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("e1")));
    expect_contains(interp, "help", "erfi(");
}

TEST(ReplWave266Pipeline, ProbChi2Ppf) {
    Interpreter interp;

    expect_ok(interp, "q = prob_chi2_ppf(0.95, 1)");
    ASSERT_GT(interp.state().scalars.count("q"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("q"), 3.841, 1e-2);
    expect_contains(interp, "help", "prob_chi2_ppf(");
}

TEST(ReplWave266Pipeline, GraphMinArborescence) {
    Interpreter interp;

    // Directed chain 0 -> 1 -> 2; minimum arborescence from root 0
    expect_ok(interp, "G = [0, 1, 0; 0, 0, 1; 0, 0, 0]");
    expect_ok(interp, "ma = graph_min_arborescence(G, 0)");
    const bool has_scalar = interp.state().scalars.count("ma") > 0u;
    const bool has_matrix = interp.state().matrices.count("ma") > 0u;
    EXPECT_TRUE(has_scalar || has_matrix);
    expect_contains(interp, "help", "graph_min_arborescence(");
}

TEST(ReplWave266Pipeline, ImageSobelX) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "sx = sobel_x(M)");
    ASSERT_GT(interp.state().matrices.count("sx"), 0u);
    EXPECT_GE(interp.state().matrices.at("sx").rows(), 1u);
    expect_contains(interp, "help", "sobel_x(");
}

TEST(ReplWave266Pipeline, StatsKsNorm) {
    Interpreter interp;

    expect_ok(interp, "x = [0; 0.1; -0.1; 0.05; -0.05]");
    expect_ok(interp, "ks = stats_ks_norm(x)");
    const bool has_scalar = interp.state().scalars.count("ks") > 0u;
    const bool has_matrix = interp.state().matrices.count("ks") > 0u;
    EXPECT_TRUE(has_scalar || has_matrix);
    expect_contains(interp, "help", "stats_ks_norm(");
}

// ---- Wave 265 regressions ----

TEST(ReplWave266Pipeline, PolyShiftRegression) {
    Interpreter interp;

    expect_ok(interp, "p = [0; 0; 1]");
    expect_ok(interp, "s = poly_shift(p, 1)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("s")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("s")(1, 0), -2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("s")(2, 0), 1.0, 1e-9);
    expect_contains(interp, "help", "poly_shift(p,a)");
}

TEST(ReplWave266Pipeline, ControlSeriesRegression) {
    Interpreter interp;

    expect_ok(interp, "num1 = [1]");
    expect_ok(interp, "den1 = [1; 1]");
    expect_ok(interp, "num2 = [2]");
    expect_ok(interp, "den2 = [2; 1]");
    expect_ok(interp, "ser = control_series(num1, den1, num2, den2)");
    ASSERT_GT(interp.state().matrices.count("ser"), 0u);
    EXPECT_GE(interp.state().matrices.at("ser").rows(), 1u);
    expect_contains(interp, "help", "control_series(num1,den1,num2,den2)");
}

TEST(ReplWave266Pipeline, BesselKRegression) {
    Interpreter interp;

    expect_ok(interp, "k0 = bessel_k(0, 1)");
    ASSERT_GT(interp.state().scalars.count("k0"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("k0"), 0.421024438240708, 1e-6);
    expect_contains(interp, "help", "bessel_k(");
}

TEST(ReplWave266Pipeline, InfoNormalizedEntropyRegression) {
    Interpreter interp;

    expect_ok(interp, "p = [0.25; 0.25; 0.25; 0.25]");
    expect_ok(interp, "hn = info_normalized_entropy(p)");
    ASSERT_GT(interp.state().scalars.count("hn"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("hn"), 1.0, 1e-9);
    expect_contains(interp, "help", "info_normalized_entropy(p)");
}
