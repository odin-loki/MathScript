
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

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

TEST(IntegrationMl,  MlMetricsGboostIsoAggloTsneGolombTail18) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_confusion_matrix(p,t");
    expect_contains(interp, "help", "ml_roc_curve(p,t)");
    expect_contains(interp, "help", "ml_precision_recall_curve(p,t)");
    expect_contains(interp, "help", "ml_gradient_boosting_fit(X,y");
    expect_contains(interp, "help", "ml_isolation_forest_fit");
    expect_contains(interp, "help", "ml_agglomerative_fit");
    expect_contains(interp, "help", "ml_tsne_fit");
    expect_contains(interp, "help", "golomb_rice_encode_vec");

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

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(IntegrationMl,  NumthyGcdScalar) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}
