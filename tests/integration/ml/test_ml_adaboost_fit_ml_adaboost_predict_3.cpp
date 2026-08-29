
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

TEST(IntegrationMl,  MlAdaboostGmmTail21) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_adaboost_fit");
    expect_contains(interp, "help", "ml_gmm_fit");

    expect_ok(interp, "X = [0,0; 0,1; 1,0; 1,1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
    EXPECT_GE(interp.state().matrices.at("gmm_p").cols(), 2u);
}

TEST(IntegrationMl,  Theta2Scalar) {
    Interpreter interp;
    expect_ok(interp, "th2 = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th2"), ms::theta2(0.2, 0.1), 1e-8);
}
