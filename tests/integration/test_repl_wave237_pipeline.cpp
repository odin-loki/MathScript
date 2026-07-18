// MathScript Integration Tests: REPL Interpreter – Wave 237 Bindings Pipeline
//
// Wired: finance_sabr_call, graph_bridges, graph_min_cut, graph_transitive_closure,
//        control_lqe, scharr, roberts.

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

void expect_error(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    EXPECT_FALSE(result.has_value()) << cmd;
}

} // namespace

TEST(ReplWave237Pipeline, FinanceSabrCallBinding) {
    Interpreter interp;

    expect_ok(interp, "c = finance_sabr_call(100, 100, 1, 0.05, 0.20, 0.5, -0.30, 0.40)");
    ASSERT_GT(interp.state().scalars.count("c"), 0u);
    EXPECT_GT(interp.state().scalars.at("c"), 0.0);
    EXPECT_LT(interp.state().scalars.at("c"), 100.0);

    expect_error(interp, "finance_sabr_call(100, 100, 1, 0.05, 0.20, 0.5, -0.30)");
    expect_contains(interp, "help", "finance_sabr_call");
}

TEST(ReplWave237Pipeline, GraphBridgesMinCutTransitiveClosureBindings) {
    Interpreter interp;

    // Barbell: two triangles joined by bridge edge 2-3.
    expect_ok(interp,
              "A = [0, 1, 1, 0, 0, 0; "
              "1, 0, 1, 0, 0, 0; "
              "1, 1, 0, 1, 0, 0; "
              "0, 0, 1, 0, 1, 1; "
              "0, 0, 0, 1, 0, 1; "
              "0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "br = graph_bridges(A)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("br")(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("br")(0, 1), 3.0, 1e-9);

    // Directed reachability: 0->1->2 implies closure[0,2]=1.
    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-9);

    // Min cut on a simple s-t cut with capacity 8.
    expect_ok(interp, "F = [0, 8, 0; 0, 0, 8; 0, 0, 0]");
    expect_ok(interp, "cut = graph_min_cut(F, 0, 2)");
    ASSERT_GT(interp.state().scalars.count("cut"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("cut"), 8.0, 1e-6);

    expect_contains(interp, "help", "graph_bridges(A)");
    expect_contains(interp, "help", "graph_transitive_closure(A)");
    expect_contains(interp, "help", "graph_min_cut(A,source,sink)");
}

TEST(ReplWave237Pipeline, ControlLqeBinding) {
    Interpreter interp;

    expect_ok(interp, "A = [-2, 1; 0, -3]");
    expect_ok(interp, "C = [1, 0]");
    expect_ok(interp, "Q = [1, 0; 0, 1]");
    expect_ok(interp, "R = [1]");
    expect_ok(interp, "L = control_lqe(A, C, Q, R)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 1u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("L")(0, 0)));

    expect_error(interp, "control_lqe(A, C, Q)");
    expect_contains(interp, "help", "control_lqe(A,C,Q,R)");
}

TEST(ReplWave237Pipeline, ImageScharrRobertsBindings) {
    Interpreter interp;

    // Vertical step: strong response near the edge.
    expect_ok(interp,
              "img = [0, 0, 0, 0, 0; "
              "0, 0, 0, 0, 0; "
              "0, 0, 0, 1, 1; "
              "0, 0, 0, 1, 1; "
              "0, 0, 0, 1, 1]");
    expect_ok(interp, "sc = scharr(img)");
    ASSERT_GT(interp.state().matrices.count("sc"), 0u);
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);

    expect_ok(interp, "rb = roberts(img)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);
    EXPECT_GT(interp.state().matrices.at("rb")(2, 2), 0.0);

    expect_error(interp, "scharr()");
    expect_contains(interp, "help", "scharr(M)");
    expect_contains(interp, "help", "roberts(M)");
}
