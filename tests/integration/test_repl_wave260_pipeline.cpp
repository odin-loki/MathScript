// MathScript Integration Tests: REPL Interpreter – Wave 260 Pipeline
//
// control_step_response / control_impulse_response REPL bindings.

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error";
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplWave260Pipeline, ControlStepResponse) {
    Interpreter interp;

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 100)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_EQ(step.rows(), 100u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.08);

    expect_contains(interp, "help", "control_step_response(num,den[,t_end[,n_pts]])");
}

TEST(ReplWave260Pipeline, ControlImpulseResponse) {
    Interpreter interp;

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 100)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    const auto& imp = interp.state().matrices.at("imp");
    EXPECT_EQ(imp.cols(), 2u);
    EXPECT_EQ(imp.rows(), 100u);
    EXPECT_NEAR(imp(imp.rows() - 1, 1), std::exp(-5.0), 0.08);

    expect_contains(interp, "help", "control_impulse_response(num,den[,t_end[,n_pts]])");
}
