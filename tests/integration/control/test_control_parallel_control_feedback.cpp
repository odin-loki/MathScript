
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

TEST(IntegrationControl,  ParallelFeedback) {
    Interpreter interp;
    expect_contains(interp, "help", "control_parallel(num1,den1,num2,den2)");
    expect_contains(interp, "help", "control_feedback(numG,denG,numH,denH");

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    EXPECT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(IntegrationControl,  BesselKScalar) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1), 1e-8);
}
