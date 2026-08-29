
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

TEST(IntegrationMl,  MlMetricsTail11) {
    Interpreter interp;

    expect_contains(interp, "help", "ml_confusion_matrix(p,t");
    expect_contains(interp, "help", "ml_roc_curve(p,t)");
    expect_contains(interp, "help", "ml_precision_recall_curve(p,t)");

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("cm")(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("cm")(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("cm")(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("cm")(1, 1), 2.0, 1e-9);

    expect_ok(interp, "pred_p = [1;1;1;1;1;0;0;0;0;0]");
    expect_ok(interp, "true_p = [1;1;1;1;1;0;0;0;0;0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("roc")(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("roc")(0, 2), 0.0, 1e-9);
    const size_t roc_last = interp.state().matrices.at("roc").rows() - 1;
    EXPECT_NEAR(interp.state().matrices.at("roc")(roc_last, 1), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("roc")(roc_last, 2), 1.0, 1e-9);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);
    EXPECT_GE(interp.state().matrices.at("pr").rows(), 2u);
}

TEST(IntegrationMl,  HermiteHScalar) {
    Interpreter interp;

    expect_ok(interp, "h = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h"), ms::hermite_h(2, 0.5), 1e-8);
}
