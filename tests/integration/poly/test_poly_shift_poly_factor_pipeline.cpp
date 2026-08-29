
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

TEST(IntegrationPoly,  PolyShift) {
    Interpreter interp;

    // x^2 shifted by 1 -> (x-1)^2 = 1 - 2x + x^2
    expect_ok(interp, "p = [0; 0; 1]");
    expect_ok(interp, "s = poly_shift(p, 1)");
    ASSERT_GT(interp.state().matrices.count("s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s").cols(), 1u);
    EXPECT_GE(interp.state().matrices.at("s").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("s")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("s")(1, 0), -2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("s")(2, 0), 1.0, 1e-9);
    expect_contains(interp, "help", "poly_shift(p,a)");
}

TEST(IntegrationPoly,  PolyFactor) {
    Interpreter interp;

    // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
    expect_ok(interp, "p = [-6; 11; -6; 1]");
    expect_ok(interp, "f = poly_factor(p)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GE(interp.state().matrices.at("f").rows(), 1u);
    EXPECT_GE(interp.state().matrices.at("f").cols(), 1u);
    expect_contains(interp, "help", "poly_factor(p)");
}

TEST(IntegrationPoly,  ControlSeries) {
    Interpreter interp;

    // 1/(s+1) * 2/(s+2) = 2/(s^2+3s+2)
    expect_ok(interp, "num1 = [1]");
    expect_ok(interp, "den1 = [1; 1]");
    expect_ok(interp, "num2 = [2]");
    expect_ok(interp, "den2 = [2; 1]");
    expect_ok(interp, "ser = control_series(num1, den1, num2, den2)");
    ASSERT_GT(interp.state().matrices.count("ser"), 0u);
    EXPECT_GE(interp.state().matrices.at("ser").rows(), 1u);
    expect_contains(interp, "help", "control_series(num1,den1,num2,den2)");
}

TEST(IntegrationPoly,  FinanceMertonImpliedAssetParams) {
    Interpreter interp;

    // Observable equity from a healthy firm; implied asset params should be finite.
    expect_ok(interp, "m = finance_merton_implied_asset_params(120, 0.25, 100, 0.05, 1.0)");
    const bool has_scalar = interp.state().scalars.count("m") > 0u;
    const bool has_matrix = interp.state().matrices.count("m") > 0u;
    EXPECT_TRUE(has_scalar || has_matrix);
    if (has_scalar) {
        EXPECT_TRUE(std::isfinite(interp.state().scalars.at("m")));
    } else {
        EXPECT_GE(interp.state().matrices.at("m").rows(), 1u);
    }
    expect_contains(interp, "help", "finance_merton_implied_asset_params(");
}

TEST(IntegrationPoly,  ComboPrevPerm) {
    Interpreter interp;

    // prev of [1,3,2] is [1,2,3]
    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pp").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("pp")(1, 0), 2.0, 1e-9);
    expect_contains(interp, "help", "combo_prev_perm(v)");
}

TEST(IntegrationPoly,  BesselK) {
    Interpreter interp;

    expect_ok(interp, "k0 = bessel_k(0, 1)");
    ASSERT_GT(interp.state().scalars.count("k0"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("k0"), 0.421024438240708, 1e-6);
    expect_contains(interp, "help", "bessel_k(");
}

TEST(IntegrationPoly,  MatrixRank) {
    Interpreter interp;

    expect_ok(interp, "A = eye(3)");
    expect_ok(interp, "rk = matrix_rank(A)");
    ASSERT_GT(interp.state().scalars.count("rk"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("rk"), 3.0, 1e-9);
    expect_contains(interp, "help", "matrix_rank(A");
}

TEST(IntegrationPoly,  GraphConnectedComponents) {
    Interpreter interp;

    // Two disjoint edges: {0,1} and {2,3}
    expect_ok(interp, "G = [0, 1, 0, 0; 1, 0, 0, 0; 0, 0, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "cc = graph_connected_components(G)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
    EXPECT_GE(interp.state().matrices.at("cc").rows(), 1u);
    expect_contains(interp, "help", "graph_connected_components(A)");
}

TEST(IntegrationPoly,  InfoNormalizedEntropy) {
    Interpreter interp;

    expect_ok(interp, "p = [0.25; 0.25; 0.25; 0.25]");
    expect_ok(interp, "hn = info_normalized_entropy(p)");
    ASSERT_GT(interp.state().scalars.count("hn"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("hn"), 1.0, 1e-9);
    expect_contains(interp, "help", "info_normalized_entropy(p)");
}

// ---- Wave 264 regressions ----

TEST(IntegrationPoly,  PolyRootsRegression) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "r = poly_roots(p)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("r").cols(), 2u);
    expect_contains(interp, "help", "poly_roots(p)");
}

TEST(IntegrationPoly,  ControlTf2ssRegression) {
    Interpreter interp;

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [1; 3; 2]");
    expect_ok(interp, "ssA = control_tf2ss(num, den)");
    ASSERT_GT(interp.state().matrices.count("ssA"), 0u);
    EXPECT_GE(interp.state().matrices.at("ssA").rows(), 2u);
    expect_contains(interp, "help", "control_tf2ss(num,den)");
}

TEST(IntegrationPoly,  InfoChannelCapacityRegression) {
    Interpreter interp;

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "cap = info_channel_capacity(W)");
    ASSERT_GT(interp.state().scalars.count("cap"), 0u);
    EXPECT_GT(interp.state().scalars.at("cap"), 0.0);
    EXPECT_LT(interp.state().scalars.at("cap"), 1.0 + 1e-6);
    expect_contains(interp, "help", "info_channel_capacity(W)");
}

TEST(IntegrationPoly,  BesselYRegression) {
    Interpreter interp;

    expect_ok(interp, "y0 = bessel_y(0, 1)");
    ASSERT_GT(interp.state().scalars.count("y0"), 0u);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("y0")));
    expect_contains(interp, "help", "bessel_y(");
}
