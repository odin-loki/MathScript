
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

TEST(IntegrationMl,  MlTreeEnsembleTail11) {
    Interpreter interp;

    expect_contains(interp, "help", "ml_decision_tree_fit(X,y");
    expect_contains(interp, "help", "ml_random_forest_fit(X,y");
    expect_contains(interp, "help", "ml_adaboost_fit(X,y");

    expect_ok(interp, "X = [0,0; 0,1; 1,0; 1,1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);
    int dt_correct = 0;
    for (size_t i = 0; i < 4; ++i) {
        if (std::abs(interp.state().matrices.at("dt_p")(i, 0) -
                     interp.state().matrices.at("y")(i, 0)) < 0.5) {
            ++dt_correct;
        }
    }
    EXPECT_GE(dt_correct, 3);

    expect_ok(interp, "Rfx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
    int rf_correct = 0;
    for (size_t i = 0; i < 6; ++i) {
        if (std::abs(interp.state().matrices.at("rf_p")(i, 0) -
                     interp.state().matrices.at("Rfy")(i, 0)) < 0.5) {
            ++rf_correct;
        }
    }
    EXPECT_GE(rf_correct, 5);

    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);
}

TEST(IntegrationMl,  HermiteHfSpheroidalScalar) {
    Interpreter interp;

    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);

    expect_ok(interp, "sl = spheroidal_lambda(1, 0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::spheroidal_lambda(1, 0, 0.1), 1e-6);
}
