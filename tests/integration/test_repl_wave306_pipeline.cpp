// MathScript Integration Tests: REPL Interpreter – Wave 306 Pipeline
//
// Wave 306 REPL smoke: ML/finance tail11 extensions, Lambert W scalar validation.

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/special/special.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error: "
                                    << (result ? *result : "unknown");
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplWave306Pipeline, MlFinanceTail11) {
    Interpreter interp;

    expect_contains(interp, "help", "ml_logistic_fit(X,y)");
    expect_contains(interp, "help", "ml_lasso_fit(X,y,alpha)");
    expect_contains(interp, "help", "finance_merton_implied_asset_params");

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);
}

TEST(ReplWave306Pipeline, LambertWScalar) {
    Interpreter interp;

    const double w_ref = ms::lambert_w(0, 1.0);
    expect_ok(interp, "w = lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("w"), w_ref, 1e-9);

    expect_ok(interp, "wm = lambert_w(-1, -0.2)");
    EXPECT_NEAR(interp.state().scalars.at("wm"), ms::lambert_w(-1, -0.2), 1e-9);
}
