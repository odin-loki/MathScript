// MathScript Integration Tests: REPL Interpreter – Wave 260 Pipeline
//
// control_step_response / control_impulse_response REPL bindings.
// bidiag / solve_sylvester / minres REPL bindings.

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

TEST(ReplWave260Pipeline, BidiagAssignAndMulti) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "B = bidiag(A)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("B").cols(), 2u);

    expect_ok(interp, "U, Bd, V = bidiag(A)");
    ASSERT_GT(interp.state().matrices.count("U"), 0u);
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.count("V"), 0u);

    expect_contains(interp, "help", "U, B, V = bidiag(A)");
}

TEST(ReplWave260Pipeline, SolveSylvester) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0; 0, 2]");
    expect_ok(interp, "B = [3, 0; 0, 4]");
    // A*X + X*B with X=[1,2;3,4] => C=[4,10;15,24]
    expect_ok(interp, "C = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(A, B, C)");
    ASSERT_GT(interp.state().matrices.count("X"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 1), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("X")(1, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("X")(1, 1), 4.0, 1e-8);
    expect_contains(interp, "help", "solve_sylvester(A,B,C)");
}

TEST(ReplWave260Pipeline, Minres) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "b = [1; 2; 3]");
    expect_ok(interp, "x = minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 2.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(2, 0), 3.0, 1e-6);
    expect_contains(interp, "help", "minres(A");
}
