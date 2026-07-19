// MathScript Integration Tests: REPL Interpreter – Wave 267 Pipeline
//
// Wave 267 REPL smoke: mathieu_a, painleve3, ode_rk23, ml_pca_fit,
// geo_upper_hull, stats_max_value.
// Regressions: Wave 266 poly_cheb_expand, erfi, stats_ks_norm.
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

TEST(ReplWave267Pipeline, SpecialMathieuA) {
    Interpreter interp;

    expect_ok(interp, "a = mathieu_a(1, 0.1)");
    ASSERT_GT(interp.state().scalars.count("a"), 0u);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("a")));
    expect_contains(interp, "help", "mathieu_a(");
}

TEST(ReplWave267Pipeline, SpecialPainleve3) {
    Interpreter interp;

    expect_ok(interp, "p3 = painleve3(0.5, 0.5, -0.1, 0.5, 0.3)");
    ASSERT_GT(interp.state().scalars.count("p3"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("p3"), 1.398748842793728, 1e-2);
    expect_contains(interp, "help", "painleve3(");
}

TEST(ReplWave267Pipeline, OdeRk23) {
    Interpreter interp;

    // y' = y from t=0..1; adaptive RK23 should reach exp(1) within tolerance
    expect_ok(interp, "ode_rk23(\"y\", 0, 1, 1, 1e-6, 1e-9)");
    expect_contains(interp, "help", "ode_rk23(");
}

TEST(ReplWave267Pipeline, MlPcaFit) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 0; 0, 1; 1, 1; 2, 0]");
    expect_ok(interp, "model = ml_pca_fit(X, 2)");
    const bool has_scalar = interp.state().scalars.count("model") > 0u;
    const bool has_matrix = interp.state().matrices.count("model") > 0u;
    EXPECT_TRUE(has_scalar || has_matrix);
    expect_contains(interp, "help", "ml_pca_fit(");
}

TEST(ReplWave267Pipeline, GeoUpperHull) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 1, 1; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "uh = geo_upper_hull(P)");
    ASSERT_GT(interp.state().matrices.count("uh"), 0u);
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("uh").cols(), 2u);
    expect_contains(interp, "help", "geo_upper_hull(");
}

TEST(ReplWave267Pipeline, StatsMaxValue) {
    Interpreter interp;

    expect_ok(interp, "mx = stats_max_value([3; 1; 4; 1; 5; 9; 2])");
    ASSERT_GT(interp.state().scalars.count("mx"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("mx"), 9.0, 1e-9);
    expect_contains(interp, "help", "stats_max_value(");
}

// ---- Wave 266 regressions ----

TEST(ReplWave267Pipeline, PolyChebExpandRegression) {
    Interpreter interp;

    expect_ok(interp, "p = [1; -2; 0; 1]");
    expect_ok(interp, "c = poly_cheb_expand(p, 3)");
    ASSERT_GT(interp.state().matrices.count("c"), 0u);
    EXPECT_GE(interp.state().matrices.at("c").rows(), 1u);
    expect_ok(interp, "v = poly_cheb_eval(c, 0)");
    ASSERT_GT(interp.state().scalars.count("v"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("v"), 1.0, 1e-6);
    expect_contains(interp, "help", "poly_cheb_expand(");
}

TEST(ReplWave267Pipeline, SpecialErfiRegression) {
    Interpreter interp;

    expect_ok(interp, "e0 = erfi(0)");
    ASSERT_GT(interp.state().scalars.count("e0"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("e0"), 0.0, 1e-12);
    expect_ok(interp, "e1 = erfi(1)");
    ASSERT_GT(interp.state().scalars.count("e1"), 0u);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("e1")));
    expect_contains(interp, "help", "erfi(");
}

TEST(ReplWave267Pipeline, StatsKsNormRegression) {
    Interpreter interp;

    expect_ok(interp, "x = [0; 0.1; -0.1; 0.05; -0.05]");
    expect_ok(interp, "ks = stats_ks_norm(x, 0, 1)");
    ASSERT_GT(interp.state().scalars.count("ks"), 0u);
    EXPECT_GE(interp.state().scalars.at("ks"), 0.0);
    EXPECT_LE(interp.state().scalars.at("ks"), 1.0);
    expect_contains(interp, "help", "stats_ks_norm(");
}
