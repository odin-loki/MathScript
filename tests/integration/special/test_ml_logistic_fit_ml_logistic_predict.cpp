
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

TEST(IntegrationSpecial,  LogisticFitPredictTail28) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_logistic_fit");
    expect_contains(interp, "help", "ml_logistic_predict");

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(IntegrationSpecial,  BesselJScalar) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-8);
}
