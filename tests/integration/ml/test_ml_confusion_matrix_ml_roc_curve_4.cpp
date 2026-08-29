
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

TEST(IntegrationMl,  MlMetricsGboostTail21) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_confusion_matrix");
    expect_contains(interp, "help", "ml_roc_curve");
    expect_contains(interp, "help", "ml_precision_recall_curve");
    expect_contains(interp, "help", "ml_gradient_boosting_fit");

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(IntegrationMl,  NumthyGcdScalar) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}
