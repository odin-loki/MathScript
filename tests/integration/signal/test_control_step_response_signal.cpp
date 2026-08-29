
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

TEST(IntegrationSignal,  StepConv2Tail14) {
    Interpreter interp;
    expect_contains(interp, "help", "control_step_response");
    expect_contains(interp, "help", "signal_conv2");

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    EXPECT_EQ(interp.state().matrices.at("step").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("step").cols(), 2u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 3u);
}

TEST(IntegrationSignal,  WeberEScalar) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(1, 1.0), 1e-8);
}
