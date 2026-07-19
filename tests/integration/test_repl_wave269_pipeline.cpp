// MathScript Integration Tests: REPL Interpreter – Wave 269 Pipeline
//
// Wave 269 REPL smoke (pipeline-only; no feature bindings in this wave):
// cross-check stable Wave 268 APIs that Wave 269 feature branches will extend.
//
// Covered: ml_standard_scaler_fit, pde_heat_1d_cn, brentq, fem_poisson1d,
// arithmetic_encode_vec.
//
// Wave 269 APIs (deferred — add tests when bindings land):
//   ml_standard_scaler_transform, ml_minmax_scaler_fit, ml_minmax_scaler_transform,
//   ml_train_test_split, ml_roc_auc, ml_average_precision,
//   pde_wave_2d, pde_advection_1d_lax_wendroff, pde_reaction_diffusion_1d,
//   ml_qda_fit, ans_encode_vec, ans_decode_vec.

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
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

double parse_scalar_x_opt(const std::string& text) {
    const auto pos = text.find("x_opt = ");
    if (pos == std::string::npos) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::stod(text.substr(pos + 8));
}

void expect_near_in_output(Interpreter& interp, const std::string& cmd, double expected,
                           double tol) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    const double x_opt = parse_scalar_x_opt(*result);
    ASSERT_FALSE(std::isnan(x_opt)) << cmd << " output: " << *result;
    EXPECT_NEAR(x_opt, expected, tol) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplWave269Pipeline, MlStandardScalerFit) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    ASSERT_GT(interp.state().matrices.count("ss_m"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ss_m").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("ss_m").cols(), 2u);
    expect_contains(interp, "help", "ml_standard_scaler_fit(");
}

TEST(ReplWave269Pipeline, PdeHeat1dCn) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x0, 0.1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("hcn").cols(), 1u);
    for (std::size_t i = 0; i < 5u; ++i) {
        EXPECT_TRUE(std::isfinite(interp.state().matrices.at("hcn")(i, 0)));
    }
    expect_contains(interp, "help", "pde_heat_1d_cn(");
}

TEST(ReplWave269Pipeline, Brentq) {
    Interpreter interp;

    // (x-2)(x+1) has root x=2 on [0, 5]
    expect_near_in_output(interp, R"cmd(brentq("(x0-2)*(x0+1)", 0, 5))cmd", 2.0, 1e-6);
    expect_contains(interp, "help", "brentq(");
}

TEST(ReplWave269Pipeline, FemPoisson1d) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(8)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GE(interp.state().matrices.at("u1").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("u1").cols(), 1u);
    EXPECT_GT(interp.state().matrices.at("u1")(4, 0), 0.0);
    expect_contains(interp, "help", "fem_poisson1d(");
}

TEST(ReplWave269Pipeline, ArithmeticEncodeVec) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "E = arithmetic_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_GE(interp.state().matrices.at("E").rows(), 1u);
    expect_contains(interp, "help", "arithmetic_encode_vec(");
}
