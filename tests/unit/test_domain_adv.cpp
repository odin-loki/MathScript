// MathScript: Advanced Domain Operation Tests
// Tests for factorial, nchoosek, gcd, Graph, graph_num_edges

#include <gtest/gtest.h>
#include <cstddef>
#include <vector>
#include "ms/domain/domain.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// factorial
// ---------------------------------------------------------------------------

TEST(DomainAdv, Factorial_Zero_IsOne) {
    EXPECT_EQ(factorial(0), 1u);
}

TEST(DomainAdv, Factorial_One_IsOne) {
    EXPECT_EQ(factorial(1), 1u);
}

TEST(DomainAdv, Factorial_SmallValues) {
    EXPECT_EQ(factorial(2), 2u);
    EXPECT_EQ(factorial(3), 6u);
    EXPECT_EQ(factorial(4), 24u);
    EXPECT_EQ(factorial(5), 120u);
}

TEST(DomainAdv, Factorial_Recursive_Property) {
    // n! = n * (n-1)!
    for (size_t n = 2; n <= 10; ++n) {
        EXPECT_EQ(factorial(n), n * factorial(n - 1)) << "at n=" << n;
    }
}

TEST(DomainAdv, Factorial_10) {
    EXPECT_EQ(factorial(10), 3628800u);
}

TEST(DomainAdv, Factorial_12) {
    EXPECT_EQ(factorial(12), 479001600u);
}

// ---------------------------------------------------------------------------
// nchoosek
// ---------------------------------------------------------------------------

TEST(DomainAdv, NChooseK_KIs0_IsOne) {
    EXPECT_EQ(nchoosek(5, 0), 1u);
    EXPECT_EQ(nchoosek(0, 0), 1u);
}

TEST(DomainAdv, NChooseK_KIsN_IsOne) {
    EXPECT_EQ(nchoosek(5, 5), 1u);
    EXPECT_EQ(nchoosek(10, 10), 1u);
}

TEST(DomainAdv, NChooseK_KIs1_IsN) {
    EXPECT_EQ(nchoosek(7, 1), 7u);
    EXPECT_EQ(nchoosek(10, 1), 10u);
}

TEST(DomainAdv, NChooseK_Symmetric) {
    // C(n, k) = C(n, n-k)
    for (size_t n = 0; n <= 10; ++n) {
        for (size_t k = 0; k <= n; ++k) {
            EXPECT_EQ(nchoosek(n, k), nchoosek(n, n - k))
                << "C(" << n << "," << k << ") != C(" << n << "," << n - k << ")";
        }
    }
}

TEST(DomainAdv, NChooseK_PascalRule) {
    // C(n+1, k) = C(n, k) + C(n, k-1)
    for (size_t n = 1; n <= 8; ++n) {
        for (size_t k = 1; k <= n; ++k) {
            EXPECT_EQ(nchoosek(n + 1, k), nchoosek(n, k) + nchoosek(n, k - 1))
                << "Pascal's rule failed at n=" << n << " k=" << k;
        }
    }
}

TEST(DomainAdv, NChooseK_SpecificValues) {
    EXPECT_EQ(nchoosek(4, 2), 6u);
    EXPECT_EQ(nchoosek(6, 3), 20u);
    EXPECT_EQ(nchoosek(10, 4), 210u);
    EXPECT_EQ(nchoosek(20, 10), 184756u);
}

// ---------------------------------------------------------------------------
// gcd
// ---------------------------------------------------------------------------

TEST(DomainAdv, GCD_WithZero_IsOther) {
    // gcd(n, 0) = n
    EXPECT_EQ(gcd(5u, 0u), 5u);
    EXPECT_EQ(gcd(0u, 7u), 7u);
}

TEST(DomainAdv, GCD_SameNumber_IsSelf) {
    EXPECT_EQ(gcd(12u, 12u), 12u);
    EXPECT_EQ(gcd(7u, 7u), 7u);
}

TEST(DomainAdv, GCD_CoPrimes_IsOne) {
    EXPECT_EQ(gcd(7u, 13u), 1u);
    EXPECT_EQ(gcd(4u, 9u), 1u);
    EXPECT_EQ(gcd(8u, 15u), 1u);
}

TEST(DomainAdv, GCD_DivisorRelationship) {
    // gcd(a, b) divides both a and b
    for (size_t a : {6u, 12u, 18u, 24u, 36u}) {
        for (size_t b : {4u, 8u, 12u, 16u}) {
            size_t g = gcd(a, b);
            EXPECT_EQ(a % g, 0u) << "gcd(" << a << "," << b << ")=" << g << " doesn't divide " << a;
            EXPECT_EQ(b % g, 0u) << "gcd(" << a << "," << b << ")=" << g << " doesn't divide " << b;
        }
    }
}

TEST(DomainAdv, GCD_Commutative) {
    EXPECT_EQ(gcd(12u, 8u), gcd(8u, 12u));
    EXPECT_EQ(gcd(100u, 75u), gcd(75u, 100u));
}

TEST(DomainAdv, GCD_SpecificValues) {
    EXPECT_EQ(gcd(12u, 8u), 4u);
    EXPECT_EQ(gcd(48u, 36u), 12u);
    EXPECT_EQ(gcd(100u, 75u), 25u);
    EXPECT_EQ(gcd(270u, 192u), 6u);
}

TEST(DomainAdv, GCD_LargeNumbers) {
    EXPECT_EQ(gcd(1000000u, 999999u), 1u);  // Consecutive integers are coprime
    EXPECT_EQ(gcd(123456u, 654321u), 3u);
}

// ---------------------------------------------------------------------------
// Graph and graph_num_edges
// ---------------------------------------------------------------------------

TEST(DomainAdv, Graph_Empty_HasZeroEdges) {
    Graph g;
    g.n = 0;
    EXPECT_EQ(graph_num_edges(g), 0u);
}

TEST(DomainAdv, Graph_SingleNode_HasZeroEdges) {
    Graph g;
    g.n = 1;
    g.adj.resize(1);
    EXPECT_EQ(graph_num_edges(g), 0u);
}

TEST(DomainAdv, Graph_TwoNodes_OneEdge) {
    Graph g;
    g.n = 2;
    g.adj.resize(2);
    g.adj[0].push_back(1);
    g.adj[1].push_back(0);  // bidirectional
    EXPECT_EQ(graph_num_edges(g), 1u);
}

TEST(DomainAdv, Graph_Complete_K4) {
    // K4 has 4*3/2 = 6 edges (bidirectional storage, total 12 adj entries)
    Graph g;
    g.n = 4;
    g.adj.resize(4);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            if (i != j) g.adj[i].push_back(j);
    size_t edges = graph_num_edges(g);
    EXPECT_EQ(edges, 6u);
}

TEST(DomainAdv, Graph_Chain_N5) {
    // Chain 0-1-2-3-4 has 4 edges (bidirectional storage)
    Graph g;
    g.n = 5;
    g.adj.resize(5);
    for (size_t i = 0; i + 1 < 5; ++i) {
        g.adj[i].push_back(i + 1);
        g.adj[i + 1].push_back(i);
    }
    size_t edges = graph_num_edges(g);
    EXPECT_EQ(edges, 4u);
}

TEST(DomainAdv, Graph_Star_N5) {
    // Star with center 0 connected to 1,2,3,4 — 4 edges (bidirectional)
    Graph g;
    g.n = 5;
    g.adj.resize(5);
    for (size_t i = 1; i < 5; ++i) {
        g.adj[0].push_back(i);
        g.adj[i].push_back(0);
    }
    size_t edges = graph_num_edges(g);
    EXPECT_EQ(edges, 4u);
}

TEST(DomainAdv, Graph_IsolatedNodes_HasZeroEdges) {
    Graph g;
    g.n = 10;
    g.adj.resize(10);  // All empty adjacency lists
    EXPECT_EQ(graph_num_edges(g), 0u);
}

TEST(DomainAdv, Graph_MultipleComponents) {
    // Two separate edges: 0-1 and 2-3 (stored bidirectionally)
    Graph g;
    g.n = 4;
    g.adj.resize(4);
    g.adj[0].push_back(1);
    g.adj[1].push_back(0);
    g.adj[2].push_back(3);
    g.adj[3].push_back(2);
    size_t edges = graph_num_edges(g);
    EXPECT_EQ(edges, 2u);
}
