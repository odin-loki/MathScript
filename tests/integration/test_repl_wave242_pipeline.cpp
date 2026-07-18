// MathScript Integration Tests: REPL Interpreter – Wave 242 Graph Bindings
//
// Wired: graph_is_isomorphic, graph_hamiltonian_path, graph_tsp_heuristic.

#include <gtest/gtest.h>
#include <cmath>
#include <string>
#include <vector>

#include "ms/core/matrix.hpp"
#include "ms/interp/repl_engine.hpp"

using namespace ms;
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

double tour_cycle_length(const Matrix<double>& tour, const Matrix<double>& dist) {
    const size_t n = tour.rows();
    if (n < 2) {
        return 0.0;
    }
    double total = 0.0;
    for (size_t i = 0; i + 1 < n; ++i) {
        const auto a = static_cast<size_t>(tour(i, 0));
        const auto b = static_cast<size_t>(tour(i + 1, 0));
        total += dist(a, b);
    }
    const auto last = static_cast<size_t>(tour(n - 1, 0));
    const auto first = static_cast<size_t>(tour(0, 0));
    total += dist(last, first);
    return total;
}

} // namespace

TEST(ReplWave242Pipeline, GraphIsIsomorphicBinding) {
    Interpreter interp;

    expect_ok(interp, "T1 = [0, 1, 1; 1, 0, 1; 1, 1, 0]");
    expect_ok(interp, "T2 = [0, 1, 1; 1, 0, 1; 1, 1, 0]");
    expect_ok(interp, "iso = graph_is_isomorphic(T1, T2)");
    ASSERT_GT(interp.state().scalars.count("iso"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("iso"), 1.0, 1e-9);

    expect_ok(interp,
              "P = [0, 1, 0, 0; "
              "1, 0, 1, 0; "
              "0, 1, 0, 1; "
              "0, 0, 1, 0]");
    expect_ok(interp,
              "S = [0, 1, 1, 1; "
              "1, 0, 0, 0; "
              "1, 0, 0, 0; "
              "1, 0, 0, 0]");
    expect_ok(interp, "niso = graph_is_isomorphic(P, S)");
    ASSERT_GT(interp.state().scalars.count("niso"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("niso"), 0.0, 1e-9);

    expect_error(interp, "graph_is_isomorphic(T1)");
    expect_contains(interp, "help", "graph_is_isomorphic");
}

TEST(ReplWave242Pipeline, GraphHamiltonianPathBinding) {
    Interpreter interp;

    expect_ok(interp,
              "P = [0, 1, 0, 0; "
              "1, 0, 1, 0; "
              "0, 1, 0, 1; "
              "0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    ASSERT_GT(interp.state().matrices.count("hp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hp").cols(), 1u);
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 1, 0; 1, 0, 0; 0, 0, 0]");
    expect_error(interp, "graph_hamiltonian_path(D)");
    expect_contains(interp, "help", "graph_hamiltonian_path");
}

TEST(ReplWave242Pipeline, GraphTspHeuristicBinding) {
    Interpreter interp;

    expect_ok(interp,
              "D = [0, 3, 4; "
              "3, 0, 5; "
              "4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    ASSERT_GT(interp.state().matrices.count("tour"), 0u);
    const auto& tour = interp.state().matrices.at("tour");
    EXPECT_EQ(tour.cols(), 1u);
    EXPECT_EQ(tour.rows(), 3u);

    const auto dist = interp.state().matrices.at("D");
    EXPECT_NEAR(tour_cycle_length(tour, dist), 12.0, 1e-9);

    expect_error(interp, "graph_tsp_heuristic(D, 0)");
    expect_contains(interp, "help", "graph_tsp_heuristic");
}
