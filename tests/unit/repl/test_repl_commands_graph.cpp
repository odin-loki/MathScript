#include <algorithm>
#include <cmath>
#include <set>
#include <fstream>
#include <gtest/gtest.h>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ms/cplx/cplx.hpp"
#include "ms/control/control.hpp"
#include "ms/error/error_types.hpp"
#include "ms/finance/finance.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/ml/ml.hpp"
#include "ms/pde/pde.hpp"
#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/version.hpp"

#include "repl/repl_test_helpers.hpp"

using namespace ms::interp;

TEST(ReplCommandsTest, graph_diameter) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_diameter(A)");

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "diam = graph_diameter(Achain)");
    EXPECT_NEAR(interp.state().scalars.at("diam"), 3.0, 1e-9);
}

TEST(ReplCommandsTest, graph_spectral) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_katz_centrality(A)");
    expect_contains(interp, "help", "graph_algebraic_connectivity(A)");

    expect_ok(interp, "A = [0,1,0; 1,0,1; 0,1,0]");
    expect_ok(interp, "k = graph_katz_centrality(A)");
    ASSERT_GT(interp.state().matrices.count("k"), 0u);
    const auto& k = interp.state().matrices.at("k");
    EXPECT_EQ(k.rows(), 3u);
    EXPECT_EQ(k.cols(), 1u);
    EXPECT_TRUE(std::isfinite(k(0, 0)));
    EXPECT_TRUE(std::isfinite(k(1, 0)));
    EXPECT_TRUE(std::isfinite(k(2, 0)));

    expect_ok(interp, "ac = graph_algebraic_connectivity(A)");
    EXPECT_GT(interp.state().scalars.at("ac"), 0.0);

    expect_ok(interp, "L = graph_laplacian(A)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    // C++ adjacency_spectrum is power-iteration spectral radius (1Ã—1), not full eig.
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    ASSERT_GT(interp.state().matrices.count("spec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("spec").cols(), 1u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("spec")(0, 0)));
    EXPECT_GT(interp.state().matrices.at("spec")(0, 0), 0.0);
}

TEST(ReplCommandsTest, graph_dijkstra_bellman_ford) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_dijkstra(A,source)");
    expect_contains(interp, "help", "graph_bellman_ford(A,source)");

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "D = graph_dijkstra(A, 0)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& D = interp.state().matrices.at("D");
    EXPECT_EQ(D.rows(), 3u);
    EXPECT_EQ(D.cols(), 2u);
    EXPECT_NEAR(D(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(D(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(D(2, 0), 3.0, 1e-9);
    EXPECT_NEAR(D(0, 1), -1.0, 1e-9);
    EXPECT_NEAR(D(1, 1), 0.0, 1e-9);
    EXPECT_NEAR(D(2, 1), 1.0, 1e-9);

    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    const auto& B = interp.state().matrices.at("B");
    EXPECT_EQ(B.rows(), 3u);
    EXPECT_EQ(B.cols(), 2u);
    EXPECT_NEAR(B(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(B(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(B(2, 0), 3.0, 1e-9);
    EXPECT_NEAR(B(0, 1), -1.0, 1e-9);
    EXPECT_NEAR(B(1, 1), 0.0, 1e-9);
    EXPECT_NEAR(B(2, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, graph_structure) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_normalised_laplacian(A)");
    expect_contains(interp, "help", "graph_modularity(A,C)");
    expect_contains(interp, "help", "graph_eccentricity(A)");
    expect_contains(interp, "help", "graph_is_strongly_connected(A)");

    expect_ok(interp, "A = [0,1,0; 1,0,1; 0,1,0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    ASSERT_GT(interp.state().matrices.count("Ln"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Ln").cols(), 3u);

    expect_ok(interp, "C = [0, 1, -1; 2, -1, -1]");
    expect_ok(interp, "Q = graph_modularity(A, C)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("Q")));

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    ASSERT_GT(interp.state().matrices.count("ecc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("ecc").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("ecc")(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("ecc")(1, 0), 2.0, 1e-9);

    expect_ok(interp, "Adisc = [0, 1, 0, 0; 1, 0, 0, 0; 0, 0, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc2 = graph_eccentricity(Adisc)");
    EXPECT_NEAR(interp.state().matrices.at("ecc2")(0, 0), -1.0, 1e-9);

    expect_ok(interp, "Asc = [0, 1, 0; 0, 0, 1; 1, 0, 0]");
    expect_ok(interp, "sc = graph_is_strongly_connected(Asc)");
    EXPECT_NEAR(interp.state().scalars.at("sc"), 1.0, 1e-9);
    expect_ok(interp, "Aweak = [0, 1, 0; 0, 0, 1; 0, 0, 0]");
    expect_ok(interp, "nsc = graph_is_strongly_connected(Aweak)");
    EXPECT_NEAR(interp.state().scalars.at("nsc"), 0.0, 1e-9);
}

TEST(ReplCommandsTest, graph_radius) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_radius(A)");

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "rad = graph_radius(Achain)");
    EXPECT_NEAR(interp.state().scalars.at("rad"), 2.0, 1e-9);
}

TEST(ReplCommandsTest, graph_betweenness) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_betweenness(A)");

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
}

TEST(ReplCommandsTest, graph_closeness) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_closeness(A)");

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
    EXPECT_GT(interp.state().matrices.at("cc")(2, 0), interp.state().matrices.at("cc")(0, 0));
}

TEST(ReplCommandsTest, graph_degree_centrality) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_degree_centrality(A)");

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("dc")(0, 0), 1.0 / 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("dc")(3, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, graph_max_flow) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_max_flow(A,source,sink)");

    expect_ok(interp, "Aflow = [0, 3, 2, 0; 0, 0, 0, 2; 0, 0, 0, 3; 0, 0, 0, 0]");
    expect_ok(interp, "mf = graph_max_flow(Aflow, 0, 3)");
    EXPECT_NEAR(interp.state().scalars.at("mf"), 4.0, 1e-9);
    expect_contains(interp, "graph_max_flow(Aflow, 0, 3)", "4");
}

TEST(ReplCommandsTest, graph_min_cut) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_min_cut(A,source,sink)");

    expect_ok(interp, "Aflow = [0, 3, 2, 0; 0, 0, 0, 2; 0, 0, 0, 3; 0, 0, 0, 0]");
    expect_ok(interp, "mc = graph_min_cut(Aflow, 0, 3)");
    EXPECT_NEAR(interp.state().scalars.at("mc"), 4.0, 1e-9);
    expect_contains(interp, "graph_min_cut(Aflow, 0, 3)", "4");

    expect_error_contains(interp, "graph_min_cut(Aflow, 0, 9)", "source/sink");
    expect_ok(interp, "ns = [1, 2]");
    expect_error_contains(interp, "graph_min_cut(ns, 0, 1)", "square");
}

TEST(ReplCommandsTest, graph_bipartite_match) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_bipartite_match(A,left_size)");

    expect_ok(interp, "A = [0, 0, 1, 1; 0, 0, 1, 1; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "m = graph_bipartite_match(A, 2)");
    EXPECT_NEAR(interp.state().scalars.at("m"), 2.0, 1e-9);
    expect_contains(interp, "graph_bipartite_match(A, 2)", "2");

    expect_error_contains(interp, "graph_bipartite_match(missing, 2)", "unknown matrix");
    expect_error_contains(interp, "graph_bipartite_match(A, 1.5)", "integer");
}

TEST(ReplCommandsTest, graph_is_isomorphic) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_is_isomorphic(A,B)");

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "B = [0, 1, 1; 1, 0, 0; 1, 0, 0]");
    expect_ok(interp, "same = graph_is_isomorphic(A, A)");
    EXPECT_NEAR(interp.state().scalars.at("same"), 1.0, 1e-9);
    expect_ok(interp, "hit = graph_is_isomorphic(A, B)");
    EXPECT_NEAR(interp.state().scalars.at("hit"), 1.0, 1e-9);
    expect_ok(interp, "miss = graph_is_isomorphic(A, [0, 1, 1; 1, 0, 1; 1, 1, 0])");
    EXPECT_NEAR(interp.state().scalars.at("miss"), 0.0, 1e-9);
    expect_contains(interp, "graph_is_isomorphic(A, A)", "1");

    expect_error_contains(interp, "graph_is_isomorphic(A, missing)", "unknown matrix");
    expect_ok(interp, "ns = [1, 2]");
    expect_error_contains(interp, "graph_is_isomorphic(A, ns)", "square");
}

TEST(ReplCommandsTest, graph_chromatic_number) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_chromatic_number(A)");

    expect_ok(interp, "tri = graph_chromatic_number([0, 1, 1; 1, 0, 1; 1, 1, 0])");
    EXPECT_NEAR(interp.state().scalars.at("tri"), 3.0, 1e-9);
    expect_ok(interp, "path = graph_chromatic_number([0, 1, 0; 1, 0, 1; 0, 1, 0])");
    EXPECT_NEAR(interp.state().scalars.at("path"), 2.0, 1e-9);
    expect_contains(interp, "graph_chromatic_number([0, 1, 1; 1, 0, 1; 1, 1, 0])", "3");

    expect_error_contains(interp, "graph_chromatic_number(missing)", "unknown matrix");
    expect_ok(interp, "ns = [1, 2]");
    expect_error_contains(interp, "graph_chromatic_number(ns)", "square");
}

TEST(ReplCommandsTest, graph_k_core_subgraph) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_k_core_subgraph(A,k)");

    expect_ok(interp, "C4 = [0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0]");
    expect_ok(interp, "k2 = graph_k_core_subgraph(C4, 2)");
    ASSERT_GT(interp.state().matrices.count("k2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("k2").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("k2").cols(), 4u);

    expect_ok(interp, "k3 = graph_k_core_subgraph(C4, 3)");
    ASSERT_GT(interp.state().matrices.count("k3"), 0u);
    const auto& k3 = interp.state().matrices.at("k3");
    double k3_edges = 0.0;
    for (size_t i = 0; i < k3.rows(); ++i) {
        for (size_t j = 0; j < k3.cols(); ++j) {
            k3_edges += k3(i, j);
        }
    }
    EXPECT_NEAR(k3_edges, 0.0, 1e-9);

    expect_error_contains(interp, "bad = graph_k_core_subgraph(C4, 1.5)", "integer k");
    expect_error_contains(interp, "bad = graph_k_core_subgraph(missing, 2)", "unknown matrix");
}

TEST(ReplCommandsTest, graph_is_bipartite) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_is_bipartite(A)");

    expect_ok(interp, "bp = graph_is_bipartite([0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0])");
    EXPECT_NEAR(interp.state().scalars.at("bp"), 1.0, 1e-9);

    expect_ok(interp, "bt = graph_is_bipartite([0, 1, 1; 1, 0, 1; 1, 1, 0])");
    EXPECT_NEAR(interp.state().scalars.at("bt"), 0.0, 1e-9);
}

TEST(ReplCommandsTest, graph_is_dag) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_is_dag(A)");

    expect_ok(interp, "dag = graph_is_dag([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().scalars.at("dag"), 1.0, 1e-9);

    expect_ok(interp, "cyc = graph_is_dag([0, 1, 0; 0, 0, 1; 1, 0, 0])");
    EXPECT_NEAR(interp.state().scalars.at("cyc"), 0.0, 1e-9);

    expect_contains(interp, "graph_is_dag([0, 1, 0; 0, 0, 1; 0, 0, 0])", "1");
}

TEST(ReplCommandsTest, graph_floyd_warshall) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_floyd_warshall(A)");

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 2), 2.0, 1e-9);
}

TEST(ReplCommandsTest, graph_is_connected) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_is_connected(A)");

    expect_ok(interp, "cn = graph_is_connected([0, 1, 0; 1, 0, 1; 0, 1, 0])");
    EXPECT_NEAR(interp.state().scalars.at("cn"), 1.0, 1e-9);

    expect_ok(interp, "dc = graph_is_connected([0, 1, 0; 1, 0, 0; 0, 0, 0])");
    EXPECT_NEAR(interp.state().scalars.at("dc"), 0.0, 1e-9);
}

TEST(ReplCommandsTest, graph_mst_kruskal) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_mst_kruskal(A)");

    expect_ok(interp, "mst = graph_mst_kruskal([0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0])");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
    EXPECT_EQ(interp.state().matrices.at("mst").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("mst").cols(), 3u);
    double total_w = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        total_w += interp.state().matrices.at("mst")(i, 2);
    }
    EXPECT_NEAR(total_w, 6.0, 1e-9);
}

TEST(ReplCommandsTest, graph_mst_prim) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_mst_prim(A)");

    expect_ok(interp, "mst = graph_mst_prim([0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0])");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
    EXPECT_EQ(interp.state().matrices.at("mst").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("mst").cols(), 3u);
    double total_w = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        total_w += interp.state().matrices.at("mst")(i, 2);
    }
    EXPECT_NEAR(total_w, 6.0, 1e-9);
}

TEST(ReplCommandsTest, graph_bfs) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_bfs(A,source)");

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "ord = graph_bfs(A, 0)");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ord").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("ord")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("ord")(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("ord")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, graph_is_tree) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_is_tree(A)");

    expect_ok(interp, "tr = graph_is_tree([0, 1, 0; 1, 0, 1; 0, 1, 0])");
    EXPECT_NEAR(interp.state().scalars.at("tr"), 1.0, 1e-9);

    expect_ok(interp, "nt = graph_is_tree([0, 1, 1; 1, 0, 1; 1, 1, 0])");
    EXPECT_NEAR(interp.state().scalars.at("nt"), 0.0, 1e-9);
}

TEST(ReplCommandsTest, graph_dfs) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_dfs(A,source)");

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "ord = graph_dfs(A, 0)");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ord").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("ord")(0, 0), 0.0, 1e-9);

    expect_contains(interp, "graph_dfs(A, 0)", "0");
}

TEST(ReplCommandsTest, graph_topological_sort) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_topological_sort(A)");

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ord").rows(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("ord")(0, 0), 0.0, 1e-9);

    expect_contains(interp, "graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])", "0");
}

TEST(ReplCommandsTest, graph_greedy_colour) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_greedy_colour(A)");

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);
    EXPECT_EQ(interp.state().matrices.at("col").rows(), 4u);

    expect_ok(interp, "graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
}

TEST(ReplCommandsTest, graph_euler_circuit) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_euler_circuit(A)");

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);
    EXPECT_GT(interp.state().matrices.at("circ").rows(), 1u);

    expect_ok(interp, "graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
}

TEST(ReplCommandsTest, graph_bellman_ford_dist) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_bellman_ford_dist");

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_contains(interp, "graph_bellman_ford_dist(A, 0, 2)", "3");
}

TEST(ReplCommandsTest, graph_astar) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_astar");

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "h = [3; 2; 1; 0]");
    expect_ok(interp, "path = graph_astar(A, 0, 3, h)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("path")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("path")(3, 0), 3.0, 1e-9);

    expect_ok(interp, "graph_astar(A, 0, 3, h)");
}

TEST(ReplCommandsTest, graph_scc) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_scc(A)");

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, graph_min_arborescence) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_min_arborescence(A,root)");

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("arb").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("arb").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("arb")(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("arb")(1, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("arb")(1, 1), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("arb")(1, 2), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("arb")(2, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("arb")(2, 1), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("arb")(2, 2), 2.0, 1e-9);
}

TEST(ReplCommandsTest, graph_stats_geo_image) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "E = graph_mst_kruskal(A)");
    EXPECT_GT(interp.state().matrices.at("E").rows(), 0u);

    expect_ok(interp, "g = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(g)");
    EXPECT_EQ(interp.state().matrices.at("sw").rows(), 1u);

    expect_ok(interp, "H = geo_convex_hull([0, 0; 1, 0; 0, 1])");
    EXPECT_GE(interp.state().matrices.at("H").rows(), 3u);

    expect_ok(interp, "M = [10, 20; 30, 40]");
    expect_ok(interp, "A2 = adapthisteq(M)");
    EXPECT_EQ(interp.state().matrices.at("A2").rows(), 2u);
}

TEST(ReplCommandsTest, betweenness_closeness_degree) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);
}

TEST(ReplCommandsTest, topo_greedy_colour) {
    Interpreter interp;

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);
}

TEST(ReplCommandsTest, kcore_euler) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);
}

TEST(ReplCommandsTest, scc_louvain) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);
}

TEST(ReplCommandsTest, floyd_dijkstra) {
    Interpreter interp;

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "P = graph_dijkstra(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 3u);
}

TEST(ReplCommandsTest, biconnected_eulerian) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigenvector_katz) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_mst) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
    expect_ok(interp, "mst2 = graph_mst_prim(W)");
    ASSERT_GT(interp.state().matrices.count("mst2"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_2) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_2) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_2) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_2) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_2) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_2) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_2) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_2) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_2) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_2) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_2) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_3) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_2) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_3) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_3) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_3) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_3) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_2) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_3) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_3) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_3) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_3) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_3) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_3) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_4) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_3) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_4) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_4) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_4) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_4) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_3) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_4) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_4) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_4) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_4) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_4) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_4) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_5) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_4) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_5) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_5) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_5) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_5) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_4) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_5) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_5) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_5) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_5) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_5) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_5) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_6) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_5) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_6) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_6) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_6) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_6) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_5) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_6) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_6) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_6) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_6) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_6) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_6) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_7) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_6) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_7) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_7) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_7) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_7) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_6) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_7) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_7) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_7) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_7) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_7) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_7) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_8) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_7) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_8) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_8) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_8) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_8) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_7) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_8) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_8) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_8) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_8) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_8) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_8) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_9) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_8) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_9) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_9) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_9) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_9) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_8) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_9) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_9) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_9) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_9) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_9) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_9) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_10) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_9) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_10) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_10) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_10) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_10) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_9) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_10) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_10) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_10) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_10) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_10) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_10) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_11) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_10) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_11) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_11) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_11) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_11) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_10) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_11) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_11) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_11) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_11) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_11) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_11) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_12) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_11) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_12) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_12) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_12) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_12) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_11) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_12) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_12) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_12) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_12) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_12) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_12) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_13) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_12) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_13) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_13) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_13) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_13) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_12) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_13) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_13) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_13) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_13) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_13) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_13) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_14) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_13) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_14) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_14) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_14) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_14) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_13) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_14) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_14) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_14) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_14) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_14) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_14) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_15) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_14) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_15) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_15) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_15) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_15) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_14) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_15) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_15) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_15) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_15) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_15) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_15) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_16) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_15) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_16) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_16) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_16) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_16) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_15) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_16) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_16) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_16) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_16) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_16) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_16) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_17) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_16) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_17) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_17) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_17) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_17) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_16) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_17) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_17) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_17) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_17) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_17) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_17) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_18) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_17) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_18) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_18) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_18) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_18) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_17) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_18) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_18) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_18) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_18) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_18) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_18) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_19) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_18) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_19) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_19) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_19) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_19) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_18) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_19) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_19) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_19) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_19) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_19) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_19) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_20) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_19) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_20) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_20) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_20) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_20) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_19) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_20) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_20) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_20) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_20) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_20) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_20) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_21) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_20) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_21) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_21) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_21) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_21) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_20) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_21) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_21) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_21) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_21) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_21) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_21) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_22) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_21) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_22) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_22) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_22) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_22) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_21) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_22) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_22) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_22) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_22) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_22) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_22) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_23) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_22) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_23) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_23) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_23) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_23) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_22) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_23) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_23) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_23) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_23) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_23) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_23) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_24) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_23) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_24) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_24) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_24) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_24) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_23) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_24) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_24) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_24) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_24) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_24) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_24) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_25) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_24) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_25) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_25) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_25) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_25) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_24) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_25) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_25) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_25) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_25) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_25) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_25) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_26) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_25) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_26) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_26) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_26) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_26) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_25) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_26) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_26) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_26) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_26) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_26) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_26) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_27) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_26) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_27) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_27) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_27) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_27) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_26) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_27) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_27) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_27) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_27) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_27) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_27) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_28) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_27) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_28) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_28) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_28) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_28) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_27) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_28) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_28) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_28) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_28) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_28) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_28) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_29) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_28) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_29) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_29) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_29) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_29) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_28) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_29) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_29) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_29) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_29) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_29) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_29) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_30) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_29) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_30) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_30) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_30) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_30) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_29) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_30) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_30) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_30) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_30) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_30) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_30) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_31) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_30) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_31) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_31) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_31) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_31) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_30) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_31) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_31) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_31) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_31) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_31) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_31) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_32) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_31) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_32) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_32) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_32) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_32) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_31) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, betweenness_closeness_32) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "bc = graph_betweenness(Achain)");
    EXPECT_GT(interp.state().matrices.at("bc")(1, 0), interp.state().matrices.at("bc")(0, 0));
    expect_ok(interp, "cc = graph_closeness(Achain)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
}

TEST(ReplCommandsTest, degree_topo_32) {
    Interpreter interp;

    expect_ok(interp, "Achain = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "dc = graph_degree_centrality(Achain)");
    ASSERT_GT(interp.state().matrices.count("dc"), 0u);

    expect_ok(interp, "ord = graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("ord"), 0u);
}

TEST(ReplCommandsTest, greedy_kcore_32) {
    Interpreter interp;

    expect_ok(interp, "col = graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("col"), 0u);

    expect_ok(interp, "A = [0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0]");
    expect_ok(interp, "cores = graph_k_core_decomposition(A)");
    EXPECT_EQ(interp.state().matrices.at("cores").rows(), 4u);
}

TEST(ReplCommandsTest, euler_scc_32) {
    Interpreter interp;

    expect_ok(interp, "circ = graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])");
    ASSERT_GT(interp.state().matrices.count("circ"), 0u);

    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0]");
    expect_ok(interp, "S = graph_scc(A)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, louvain_floyd_32) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "louv = graph_louvain(B)");
    EXPECT_EQ(interp.state().matrices.at("louv").rows(), 2u);

    expect_ok(interp, "D = graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, bcc_eulerian_32) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "bcc = graph_biconnected_components(P)");
    EXPECT_EQ(interp.state().matrices.at("bcc").rows(), 3u);
    expect_ok(interp, "ep = graph_eulerian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("ep").rows(), 4u);
}

TEST(ReplCommandsTest, hamiltonian_tsp_33) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "hp = graph_hamiltonian_path(P)");
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "D = [0, 3, 4; 3, 0, 5; 4, 5, 0]");
    expect_ok(interp, "tour = graph_tsp_heuristic(D)");
    EXPECT_EQ(interp.state().matrices.at("tour").rows(), 3u);
}

TEST(ReplCommandsTest, eigen_katz_32) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "ec = graph_eigenvector_centrality(A)");
    EXPECT_EQ(interp.state().matrices.at("ec").rows(), 6u);

    expect_ok(interp, "K = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "k = graph_katz_centrality(K)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
}

TEST(ReplCommandsTest, spectrum_laplacian_33) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "spec = graph_adjacency_spectrum(A)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 1u);
    expect_ok(interp, "L = graph_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 3u);
}

TEST(ReplCommandsTest, normlap_ecc_33) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "Ln = graph_normalised_laplacian(A)");
    EXPECT_EQ(interp.state().matrices.at("Ln").rows(), 3u);

    expect_ok(interp, "Achain = [0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0]");
    expect_ok(interp, "ecc = graph_eccentricity(Achain)");
    EXPECT_EQ(interp.state().matrices.at("ecc").rows(), 4u);
}

TEST(ReplCommandsTest, articulation_bridges_33) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; 0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0]");
    expect_ok(interp, "aps = graph_articulation_points(A)");
    ASSERT_GT(interp.state().matrices.count("aps"), 0u);
    expect_ok(interp, "br = graph_bridges(A)");
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 1u);
}

TEST(ReplCommandsTest, matching_closure_33) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 0]");
    expect_ok(interp, "M = graph_maximum_matching(A)");
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 1u);

    expect_ok(interp, "R = graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])");
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 2), 1.0, 1e-9);
}

TEST(ReplCommandsTest, bellman_kruskal_32) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "B = graph_bellman_ford(A, 0)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "W = [0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0]");
    expect_ok(interp, "mst = graph_mst_kruskal(W)");
    ASSERT_GT(interp.state().matrices.count("mst"), 0u);
}

TEST(ReplCommandsTest, graph_astar_execute_no_assign) {
    Interpreter interp;
    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "h = [3; 2; 1; 0]");
    expect_contains(interp, "graph_astar(A, 0, 3, h)", "path =");
    expect_error_contains(interp, "graph_astar(A, 0.5, 3, h)",
                          "integer source and target");
}

TEST(ReplCommandsTest, graph_max_flow_execute_no_assign) {
    Interpreter interp;
    expect_ok(interp, "Aflow = [0, 3, 2, 0; 0, 0, 0, 2; 0, 0, 0, 3; 0, 0, 0, 0]");
    expect_contains(interp, "graph_max_flow(Aflow, 0, 3)", "4");
    expect_error_contains(interp, "graph_max_flow(Aflow, missing, 3)",
                          "graph_max_flow");
}

TEST(ReplCommandsTest, graph_min_cut_execute_no_assign) {
    Interpreter interp;
    expect_ok(interp, "Aflow = [0, 3, 2, 0; 0, 0, 0, 2; 0, 0, 0, 3; 0, 0, 0, 0]");
    expect_contains(interp, "graph_min_cut(Aflow, 0, 3)", "4");
    expect_error_contains(interp, "graph_min_cut(Aflow, missing, 3)",
                          "graph_min_cut");
}

TEST(ReplCommandsTest, graph_bfs_execute_no_assign) {
    Interpreter interp;
    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_contains(interp, "graph_bfs(A, 0)", "order =");
    expect_error_contains(interp, "graph_bfs(A, 1.5)",
                          "non-negative integer source");
}

TEST(ReplCommandsTest, graph_katz_centrality_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_katz_centrality([0, 1, 0; 1, 0, 1; 0, 1, 0])", "katz_centrality");
    expect_error_contains(interp, "graph_katz_centrality([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_adjacency_spectrum_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_adjacency_spectrum([0, 1, 0; 1, 0, 1; 0, 1, 0])",
                    "adjacency_spectrum");
    expect_error_contains(interp, "graph_adjacency_spectrum([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_laplacian_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_laplacian([0, 1, 0; 1, 0, 1; 0, 1, 0])", "laplacian");
    expect_error_contains(interp, "graph_laplacian([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_normalised_laplacian_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_normalised_laplacian([0, 1, 0; 1, 0, 1; 0, 1, 0])",
                    "normalised_laplacian");
    expect_error_contains(interp, "graph_normalised_laplacian([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_eccentricity_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_eccentricity([0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0])",
                    "eccentricity");
    expect_error_contains(interp, "graph_eccentricity([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_articulation_points_noassign) {
    Interpreter interp;
    expect_contains(interp,
                    "graph_articulation_points([0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; "
                    "0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0])",
                    "articulation_points");
    expect_error_contains(interp, "graph_articulation_points([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_bridges_noassign) {
    Interpreter interp;
    expect_contains(interp,
                    "graph_bridges([0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; "
                    "0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0])",
                    "bridges");
    expect_error_contains(interp, "graph_bridges([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_maximum_matching_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_maximum_matching([0, 1; 1, 0])", "matching");
    expect_error_contains(interp, "graph_maximum_matching([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_transitive_closure_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_transitive_closure([0, 1, 0; 0, 0, 1; 0, 0, 0])", "reach");
    expect_error_contains(interp, "graph_transitive_closure([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_floyd_warshall_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_floyd_warshall([0, 1, 5; 0, 0, 1; 0, 0, 0])", "dist");
    expect_error_contains(interp, "graph_floyd_warshall([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_mst_kruskal_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_mst_kruskal([0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0])",
                    "mst");
    expect_error_contains(interp, "graph_mst_kruskal([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_mst_prim_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_mst_prim([0, 1, 0, 10; 1, 0, 2, 0; 0, 2, 0, 3; 10, 0, 3, 0])",
                    "mst");
    expect_error_contains(interp, "graph_mst_prim([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_topological_sort_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_topological_sort([0, 1, 1, 0; 0, 0, 0, 1; 0, 0, 0, 1; 0, 0, 0, 0])",
                    "order");
    expect_error_contains(interp, "graph_topological_sort([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_greedy_colour_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_greedy_colour([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])",
                    "colors");
    expect_error_contains(interp, "graph_greedy_colour([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_k_core_decomposition_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_k_core_decomposition([0, 1, 1, 1; 1, 0, 1, 0; 1, 1, 0, 0; 1, 0, 0, 0])",
                    "cores");
    expect_error_contains(interp, "graph_k_core_decomposition([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_euler_circuit_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_euler_circuit([0, 1, 0, 1; 1, 0, 1, 0; 0, 1, 0, 1; 1, 0, 1, 0])",
                    "circuit");
    expect_error_contains(interp, "graph_euler_circuit([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_eulerian_path_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_eulerian_path([0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0])",
                    "path");
    expect_error_contains(interp, "graph_eulerian_path([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_hamiltonian_path_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_hamiltonian_path([0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0])",
                    "path");
    expect_error_contains(interp, "graph_hamiltonian_path([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_tsp_heuristic_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_tsp_heuristic([0, 3, 4; 3, 0, 5; 4, 5, 0])", "tour");
    expect_error_contains(interp, "graph_tsp_heuristic([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_biconnected_components_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_biconnected_components([0, 1, 0, 0; 1, 0, 1, 0; 0, 1, 0, 1; 0, 0, 1, 0])",
                    "bcc");
    expect_error_contains(interp, "graph_biconnected_components([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_scc_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_scc([0, 1, 0, 0; 0, 0, 1, 0; 1, 0, 0, 0; 0, 0, 1, 0])", "scc");
    expect_error_contains(interp, "graph_scc([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_connected_components_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_connected_components([0, 1, 0, 0; 1, 0, 0, 0; 0, 0, 0, 1; 0, 0, 1, 0])",
                    "cc");
    expect_error_contains(interp, "graph_connected_components([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_louvain_noassign) {
    Interpreter interp;
    expect_contains(interp,
                    "graph_louvain([0, 1, 1, 0, 0, 0; 1, 0, 1, 0, 0, 0; 1, 1, 0, 1, 0, 0; "
                    "0, 0, 1, 0, 1, 1; 0, 0, 0, 1, 0, 1; 0, 0, 0, 1, 1, 0])",
                    "louvain");
    expect_error_contains(interp, "graph_louvain([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_eigenvector_centrality_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_eigenvector_centrality([0, 1, 0; 1, 0, 1; 0, 1, 0])",
                    "eigenvector_centrality");
    expect_error_contains(interp, "graph_eigenvector_centrality([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_pagerank_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_pagerank([0, 1, 0; 1, 0, 1; 0, 1, 0])", "pagerank");
    expect_error_contains(interp, "graph_pagerank([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_betweenness_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_betweenness([0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0])",
                    "betweenness");
    expect_error_contains(interp, "graph_betweenness([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_closeness_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_closeness([0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0])",
                    "closeness");
    expect_error_contains(interp, "graph_closeness([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_degree_centrality_noassign) {
    Interpreter interp;
    expect_contains(interp,
                    "graph_degree_centrality([0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0])",
                    "degree_centrality");
    expect_error_contains(interp, "graph_degree_centrality([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_diameter_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_diameter([0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0])",
                    "3");
    expect_error_contains(interp, "graph_diameter([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_radius_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_radius([0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0])",
                    "2");
    expect_error_contains(interp, "graph_radius([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_is_connected_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_is_connected([0, 1, 0; 1, 0, 1; 0, 1, 0])", "1");
    expect_error_contains(interp, "graph_is_connected([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_is_dag_noassign) {
    Interpreter interp;
    expect_contains(interp, "graph_is_dag([0, 1, 0; 0, 0, 1; 0, 0, 0])", "1");
    expect_error_contains(interp, "graph_is_dag([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_is_tree_noassign) {
    Interpreter interp;
    expect_ok(interp, "graph_is_tree([0, 1, 0; 1, 0, 1; 0, 1, 0])");
    expect_error_contains(interp, "graph_is_tree([1, 2])", "square");
}

TEST(ReplCommandsTest, graph_is_bipartite_noassign) {
    Interpreter interp;
    expect_ok(interp, "graph_is_bipartite([0, 1; 1, 0])");
    expect_error_contains(interp, "graph_is_bipartite([1, 2])", "square");
}
