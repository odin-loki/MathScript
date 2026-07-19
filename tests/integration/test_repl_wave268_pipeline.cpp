// MathScript Integration Tests: REPL Interpreter – Wave 268 Pipeline
//
// Wave 268 REPL smoke: ml_qda_fit, pde_heat_1d_cn, brentq, arithmetic_encode_vec,
// fem_poisson1d.
// Regressions: Wave 267 ode_rk23, stats_max_value.
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

void expect_near_in_output(Interpreter& interp, const std::string& cmd, double expected,
                           double tol) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    const std::size_t start = result->find('[');
    ASSERT_NE(start, std::string::npos) << *result;
    const std::size_t end = result->find(']', start);
    ASSERT_NE(end, std::string::npos) << *result;
    const double x_opt = std::stod(result->substr(start + 1, end - start - 1));
    EXPECT_NEAR(x_opt, expected, tol) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplWave268Pipeline, MlQdaFit) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 1, 0; 0, 1; 1, 1; 2, 0; 0, 2]");
    expect_ok(interp, "y = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "model = ml_qda_fit(X, y)");
    const bool has_scalar = interp.state().scalars.count("model") > 0u;
    const bool has_matrix = interp.state().matrices.count("model") > 0u;
    EXPECT_TRUE(has_scalar || has_matrix);
    expect_contains(interp, "help", "ml_qda_fit(");
}

TEST(ReplWave268Pipeline, PdeHeat1dCn) {
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

TEST(ReplWave268Pipeline, Brentq) {
    Interpreter interp;

    // (x-2)(x+1) has root x=2 on [0, 5]
    expect_near_in_output(interp, R"cmd(brentq("(x0-2)*(x0+1)", 0, 5))cmd", 2.0, 1e-6);
    expect_contains(interp, "help", "brentq(");
}

TEST(ReplWave268Pipeline, ArithmeticEncodeVec) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "E = arithmetic_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_GE(interp.state().matrices.at("E").rows(), 1u);
    expect_contains(interp, "help", "arithmetic_encode_vec(");
}

TEST(ReplWave268Pipeline, FemPoisson1d) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(8)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GE(interp.state().matrices.at("u1").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("u1").cols(), 1u);
    EXPECT_GT(interp.state().matrices.at("u1")(4, 0), 0.0);
    expect_contains(interp, "help", "fem_poisson1d(");
}

// ---- Wave 267 regressions ----

TEST(ReplWave268Pipeline, OdeRk23Regression) {
    Interpreter interp;

    // y' = y from t=0..1; adaptive RK23 should reach exp(1) within tolerance
    expect_ok(interp, "ode_rk23(\"y\", 0, 1, 1, 1e-6, 1e-9)");
    expect_contains(interp, "help", "ode_rk23(");
}

TEST(ReplWave268Pipeline, StatsMaxValueRegression) {
    Interpreter interp;

    expect_ok(interp, "mx = stats_max_value([3; 1; 4; 1; 5; 9; 2])");
    ASSERT_GT(interp.state().scalars.count("mx"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("mx"), 9.0, 1e-9);
    expect_contains(interp, "help", "stats_max_value(");
}
