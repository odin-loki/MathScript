
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

TEST(IntegrationSpecial,  TransposeFunmTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_mat_transpose");
    expect_contains(interp, "help", "funm");

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-9);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-6);
}

TEST(IntegrationSpecial,  LegendreQScalar) {
    Interpreter interp;
    expect_ok(interp, "lq = legendre_q(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lq"), ms::legendre_q(1, 0.5), 1e-8);
}
