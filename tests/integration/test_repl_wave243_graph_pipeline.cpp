// MathScript Integration Tests: REPL Interpreter – Wave 243 Graph Bindings
//
// Verified existing REPL wiring: graph_articulation_points (primary target),
// graph_euler_circuit (fallback when articulation_points already bound).

#include <gtest/gtest.h>
#include <cmath>
#include <set>
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

std::set<int> matrix_column_as_int_set(const ms::Matrix<double>& m) {
    std::set<int> values;
    for (size_t i = 0; i < m.rows(); ++i) {
        values.insert(static_cast<int>(m(i, 0)));
    }
    return values;
}

} // namespace

TEST(ReplWave243Pipeline, GraphArticulationPointsBinding) {
    Interpreter interp;

    // Barbell graph: two triangles joined by edge 2-3; articulation points at 2 and 3.
    expect_ok(interp,
              "A = [0,1,1,0,0,0; 1,0,1,0,0,0; 1,1,0,1,0,0; 0,0,1,0,1,1; 0,0,0,1,0,1; 0,0,0,1,1,0]");

    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    const auto& aps = interp.state().matrices.at("aps");
    EXPECT_EQ(aps.cols(), 1u);
    EXPECT_EQ(aps.rows(), 2u);
    const auto ap_set = matrix_column_as_int_set(aps);
    EXPECT_EQ(ap_set, (std::set<int>{2, 3}));

    expect_contains(interp, "graph_articulation_points(A)", "2");
    expect_contains(interp, "graph_articulation_points(A)", "3");

    // 4-cycle has no articulation points.
    expect_ok(interp, "C = [0,1,0,1; 1,0,1,0; 0,1,0,1; 1,0,1,0]");
    expect_ok(interp, "none = graph_articulation_points(C)");
    EXPECT_EQ(interp.state().matrices.at("none").rows(), 0u);

    expect_error(interp, "graph_articulation_points(A, 0)");
    expect_contains(interp, "help", "graph_articulation_points(A)");
}

TEST(ReplWave243Pipeline, GraphEulerCircuitBinding) {
    Interpreter interp;

    expect_ok(interp, "K4 = [0,1,0,1; 1,0,1,0; 0,1,0,1; 1,0,1,0]");
    expect_ok(interp, "circ = graph_euler_circuit(K4)");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);
    const auto& circ = interp.state().matrices.at("circ");
    EXPECT_EQ(circ.cols(), 1u);
    EXPECT_GT(circ.rows(), 1u);
    EXPECT_NEAR(circ(0, 0), circ(circ.rows() - 1, 0), 1e-9);

    expect_contains(interp, "graph_euler_circuit(K4)", "circuit");

    expect_error(interp, "graph_euler_circuit(K4, 0)");
    expect_contains(interp, "help", "graph_euler_circuit(A)");
}
