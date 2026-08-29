
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

TEST(IntegrationMl,  TreeForest) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_decision_tree_fit");
    expect_contains(interp, "help", "ml_random_forest_fit");

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(IntegrationMl,  Theta1Scalar) {
    Interpreter interp;
    expect_ok(interp, "th1 = theta1(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th1"), ms::theta1(0.2, 0.1), 1e-8);
}
