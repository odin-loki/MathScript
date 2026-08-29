
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

TEST(IntegrationMl,  LinearRidge) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_linear_fit");
    expect_contains(interp, "help", "ml_ridge_fit");

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(IntegrationMl,  JacobiCsScalar) {
    Interpreter interp;
    expect_ok(interp, "jcs = jacobi_cs(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcs"), ms::jacobi_cs(0.5, 0.5), 1e-8);
}
