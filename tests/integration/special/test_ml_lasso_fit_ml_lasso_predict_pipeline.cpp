
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

TEST(IntegrationSpecial,  LassoFitPredictTail28) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_lasso_fit");
    expect_contains(interp, "help", "ml_lasso_predict");

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(IntegrationSpecial,  BesselIScalar) {
    Interpreter interp;
    expect_ok(interp, "bi = bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::bessel_i(0, 1.0), 1e-8);
}
