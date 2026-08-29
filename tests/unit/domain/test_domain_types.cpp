#include <gtest/gtest.h>
#include <cmath>
#include <expected>
#include "ms/error/error_types.hpp"
#include "ms/domain/domain.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// domain.hpp — combinatorics
// ---------------------------------------------------------------------------

TEST(DomainTypesTest, factorial_base_cases) {
    EXPECT_EQ(factorial(0), 1u);
    EXPECT_EQ(factorial(1), 1u);
}

TEST(DomainTypesTest, factorial_small_values) {
    EXPECT_EQ(factorial(2),   2u);
    EXPECT_EQ(factorial(3),   6u);
    EXPECT_EQ(factorial(5), 120u);
    EXPECT_EQ(factorial(6), 720u);
}

TEST(DomainTypesTest, nchoosek_boundary_k0_and_kn) {
    EXPECT_EQ(nchoosek(5, 0), 1u);
    EXPECT_EQ(nchoosek(5, 5), 1u);
    EXPECT_EQ(nchoosek(0, 0), 1u);
}

TEST(DomainTypesTest, nchoosek_symmetry) {
    // C(n, k) == C(n, n-k)
    EXPECT_EQ(nchoosek(7, 2), nchoosek(7, 5));
    EXPECT_EQ(nchoosek(6, 3), nchoosek(6, 3));
}

TEST(DomainTypesTest, nchoosek_known_values) {
    EXPECT_EQ(nchoosek(4, 2), 6u);
    EXPECT_EQ(nchoosek(6, 3), 20u);
    EXPECT_EQ(nchoosek(10, 4), 210u);
}

TEST(DomainTypesTest, gcd_basic) {
    EXPECT_EQ(gcd(12, 8),  4u);
    EXPECT_EQ(gcd(9, 6),   3u);
    EXPECT_EQ(gcd(7, 5),   1u);
    EXPECT_EQ(gcd(100, 25), 25u);
}

TEST(DomainTypesTest, gcd_identity_cases) {
    EXPECT_EQ(gcd(7, 7), 7u);
    EXPECT_EQ(gcd(1, 1), 1u);
}

// ---------------------------------------------------------------------------
// domain.hpp — Graph
// ---------------------------------------------------------------------------

TEST(DomainTypesTest, graph_empty) {
    Graph g;
    EXPECT_EQ(g.n, 0u);
    EXPECT_EQ(graph_num_edges(g), 0u);
}

TEST(DomainTypesTest, graph_no_edges) {
    Graph g;
    g.n = 3;
    g.adj.resize(3);
    EXPECT_EQ(graph_num_edges(g), 0u);
}

TEST(DomainTypesTest, graph_triangle) {
    // 3-node undirected cycle: 0-1, 1-2, 2-0 (each edge listed both ways)
    Graph g;
    g.n = 3;
    g.adj.resize(3);
    g.adj[0] = {1, 2};
    g.adj[1] = {0, 2};
    g.adj[2] = {0, 1};
    EXPECT_EQ(graph_num_edges(g), 3u);
}

TEST(DomainTypesTest, graph_complete_k4) {
    // K4: each node connected to every other (undirected: 4*3/2 = 6 edges)
    Graph g;
    g.n = 4;
    g.adj.resize(4);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            if (i != j) g.adj[i].push_back(j);
    EXPECT_EQ(graph_num_edges(g), 6u);
}

// ---------------------------------------------------------------------------
// error_types.hpp — field values (not just format_error)
// ---------------------------------------------------------------------------

TEST(DomainTypesTest, dimension_mismatch_fields) {
    DimensionMismatch dm(2, 3, 4, 5);
    EXPECT_EQ(dm.got_rows, 2u);
    EXPECT_EQ(dm.got_cols, 3u);
    EXPECT_EQ(dm.expected_rows, 4u);
    EXPECT_EQ(dm.expected_cols, 5u);
}

TEST(DomainTypesTest, dimension_mismatch_two_arg_ctor) {
    DimensionMismatch dm(6, 7);
    EXPECT_EQ(dm.got_rows, 6u);
    EXPECT_EQ(dm.got_cols, 7u);
    EXPECT_EQ(dm.expected_rows, 0u);
    EXPECT_EQ(dm.expected_cols, 0u);
}

TEST(DomainTypesTest, singular_matrix_field) {
    SingularMatrix sm(3.14);
    EXPECT_DOUBLE_EQ(sm.condition_number, 3.14);

    SingularMatrix sm_default;
    EXPECT_DOUBLE_EQ(sm_default.condition_number, 0.0);
}

TEST(DomainTypesTest, convergence_fail_fields) {
    ConvergenceFail cf{50, 1e-4};
    EXPECT_EQ(cf.iterations, 50u);
    EXPECT_DOUBLE_EQ(cf.residual, 1e-4);
}

TEST(DomainTypesTest, domain_error_fields) {
    DomainError de{"log", "negative argument"};
    EXPECT_EQ(de.function, "log");
    EXPECT_EQ(de.reason,   "negative argument");
}

TEST(DomainTypesTest, device_error_fields) {
    DeviceError dev{42, "out of memory"};
    EXPECT_EQ(dev.code, 42);
    EXPECT_EQ(dev.msg,  "out of memory");
}

TEST(DomainTypesTest, allocation_failure_field) {
    AllocationFailure af{4096};
    EXPECT_EQ(af.requested_bytes, 4096u);
}

// ---------------------------------------------------------------------------
// Result<T> — direct std::unexpected construction
// ---------------------------------------------------------------------------

TEST(DomainTypesTest, result_holds_value) {
    Result<int> r = 99;
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 99);
}

TEST(DomainTypesTest, result_holds_unexpected_domain_error) {
    Result<double> r = std::unexpected(Error{DomainError{"sqrt", "negative"}});
    ASSERT_FALSE(r.has_value());
    const auto* de = std::get_if<DomainError>(&r.error());
    ASSERT_NE(de, nullptr);
    EXPECT_EQ(de->function, "sqrt");
}

TEST(DomainTypesTest, result_holds_unexpected_singular) {
    Result<double> r = std::unexpected(Error{SingularMatrix{1e18}});
    ASSERT_FALSE(r.has_value());
    const auto* sm = std::get_if<SingularMatrix>(&r.error());
    ASSERT_NE(sm, nullptr);
    EXPECT_DOUBLE_EQ(sm->condition_number, 1e18);
}

TEST(DomainTypesTest, result_transform_on_value) {
    Result<int> r = 4;
    auto r2 = r.transform([](int v) { return v * v; });
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(*r2, 16);
}

TEST(DomainTypesTest, result_transform_not_called_on_error) {
    Result<int> r = std::unexpected(Error{ConvergenceFail{10, 1.0}});
    bool called = false;
    auto r2 = r.transform([&](int v) { called = true; return v + 1; });
    EXPECT_FALSE(called);
    EXPECT_FALSE(r2.has_value());
}

// ---------------------------------------------------------------------------
// format_error — additional variants not covered in test_error_types.cpp
// ---------------------------------------------------------------------------

TEST(DomainTypesTest, format_device_error_not_empty) {
    const std::string msg = format_error(Error{DeviceError{1, "GPU error"}});
    EXPECT_FALSE(msg.empty());
}

TEST(DomainTypesTest, format_allocation_failure_not_empty) {
    const std::string msg = format_error(Error{AllocationFailure{8192}});
    EXPECT_FALSE(msg.empty());
}

TEST(DomainTypesTest, format_symbolic_error_not_empty) {
    const std::string msg = format_error(Error{SymbolicError{"undefined symbol"}});
    EXPECT_FALSE(msg.empty());
}

TEST(DomainTypesTest, format_io_error_not_empty) {
    const std::string msg = format_error(Error{IOError{"/tmp/foo", "not found"}});
    EXPECT_FALSE(msg.empty());
}

TEST(DomainTypesTest, format_parse_error_not_empty) {
    const std::string msg = format_error(Error{ParseError{5, 12, "unexpected token"}});
    EXPECT_FALSE(msg.empty());
}

TEST(DomainTypesTest, format_overflow_error_not_empty) {
    const std::string msg = format_error(Error{OverflowError{"add"}});
    EXPECT_FALSE(msg.empty());
}

TEST(DomainTypesTest, format_plugin_violation_not_empty) {
    const std::string msg = format_error(Error{PluginViolation{"no-eval", "line 10"}});
    EXPECT_FALSE(msg.empty());
}

TEST(DomainTypesTest, format_distributed_error_not_empty) {
    const std::string msg = format_error(Error{DistributedError{2, 13}});
    EXPECT_FALSE(msg.empty());
}
