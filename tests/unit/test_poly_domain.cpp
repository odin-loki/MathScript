#include <gtest/gtest.h>
#include <span>
#include <vector>
#include "ms/poly/poly.hpp"
#include "ms/symbolic/symbolic.hpp"
#include "ms/domain/domain.hpp"

using namespace ms;

namespace {

std::vector<double> naive_poly_eval_at(const std::vector<double>& coeffs, std::span<const double> xs) {
    std::vector<double> out(xs.size(), 0.0);
    for (size_t i = 0; i < xs.size(); ++i) {
        double value = 0.0;
        for (size_t j = coeffs.size(); j > 0; --j) {
            value = value * xs[i] + coeffs[j - 1];
        }
        out[i] = value;
    }
    return out;
}

void expect_batch_matches_naive(const std::vector<double>& coeffs, std::span<const double> xs) {
    const auto got = poly_eval_at(coeffs, xs);
    const auto expected = naive_poly_eval_at(coeffs, xs);
    ASSERT_EQ(got.size(), expected.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], expected[i], 1e-12) << "i=" << i;
    }
}

} // namespace

TEST(PolyEvalAtTest, empty_inputs) {
    EXPECT_TRUE(poly_eval_at({}, {}).empty());
    EXPECT_TRUE(poly_eval_at({1.0, 2.0}, {}).empty());
    const std::vector<double> xs{0.5, 1.5};
    expect_batch_matches_naive({}, xs);
}

TEST(PolyEvalAtTest, constant_polynomial) {
    const std::vector<double> coeffs{7.0};
    const std::vector<double> xs{-2.0, 0.0, 3.5, 100.0};
    expect_batch_matches_naive(coeffs, xs);
}

TEST(PolyEvalAtTest, linear_polynomial_scalar_path) {
    const std::vector<double> coeffs{2.0, 3.0};
    const std::vector<double> xs{0.0, 1.0, 2.0, -1.0, 0.25};
    expect_batch_matches_naive(coeffs, xs);
}

TEST(PolyEvalAtTest, cubic_polynomial_simd_batches) {
    const std::vector<double> coeffs{1.0, -1.0, 1.0, 2.0};
    const std::vector<double> xs{-1.0, 0.0, 1.0, 2.0, 3.0, 0.5, -0.5, 4.0};
    expect_batch_matches_naive(coeffs, xs);
}

TEST(PolyEvalAtTest, high_degree_many_points) {
    std::vector<double> coeffs(21);
    for (size_t i = 0; i < coeffs.size(); ++i) {
        coeffs[i] = static_cast<double>(i + 1);
    }
    std::vector<double> xs(37);
    for (size_t i = 0; i < xs.size(); ++i) {
        xs[i] = -1.0 + 0.1 * static_cast<double>(i);
    }
    expect_batch_matches_naive(coeffs, xs);
}

TEST(PolyEvalAtTest, matches_poly_eval_scalar) {
    const std::vector<double> coeffs{4.0, -3.0, 2.0, 1.0};
    const std::vector<double> xs{0.25, 1.75, -2.0, 5.0, 9.0};
    const auto batch = poly_eval_at(coeffs, xs);
    for (size_t i = 0; i < xs.size(); ++i) {
        EXPECT_NEAR(batch[i], poly_eval(coeffs, xs[i])[0], 1e-12) << "i=" << i;
    }
}

TEST(PolyEvalAtTest, single_point_and_remainder_tail) {
    const std::vector<double> coeffs{1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> single{2.5};
    expect_batch_matches_naive(coeffs, single);
    const std::vector<double> xs{0.1, 0.2, 0.3, 0.4, 0.5};
    expect_batch_matches_naive(coeffs, xs);
}

TEST(PolyTest, eval_and_deriv) {
    std::vector<double> p{1, 2, 3};
    EXPECT_NEAR(poly_eval(p, 2.0)[0], 17.0, 1e-12);
    auto dp = poly_deriv(p);
    EXPECT_NEAR(poly_eval(dp, 2.0)[0], 14.0, 1e-12);
}

TEST(PolyTest, multiply) {
    auto a = std::vector<double>{1, 1};
    auto b = std::vector<double>{1, -1};
    auto c = poly_mul(a, b);
    EXPECT_NEAR(c[0], 1.0, 1e-12);
    EXPECT_NEAR(c[1], 0.0, 1e-12);
    EXPECT_NEAR(c[2], -1.0, 1e-12);
}

TEST(PolyTest, subtract_and_add_mismatch) {
    auto a = std::vector<double>{5.0, 2.0, 1.0};
    auto b = std::vector<double>{1.0, 1.0};
    auto diff = poly_sub(a, b);
    EXPECT_NEAR(diff[0], 4.0, 1e-12);
    EXPECT_NEAR(diff[1], 1.0, 1e-12);
    EXPECT_NEAR(diff[2], 1.0, 1e-12);
    auto sum = poly_add(diff, b);
    EXPECT_NEAR(poly_eval(sum, 0.0)[0], poly_eval(a, 0.0)[0], 1e-12);
}

TEST(PolyTest, edge_empty_and_constant) {
    EXPECT_TRUE(poly_mul({}, {1.0, 2.0}).empty());
    EXPECT_NEAR(poly_eval(poly_deriv({7.0}), 5.0)[0], 0.0, 1e-12);
    EXPECT_NEAR(poly_eval({}, 99.0)[0], 0.0, 1e-12);
}

TEST(SymbolicTest, to_string) {
    auto expr = sym_add(sym_var("x"), sym_const(1.0));
    EXPECT_EQ(sym_to_string(expr), "(x + 1.000000)");
}

TEST(DomainTest, combinatorics) {
    EXPECT_EQ(nchoosek(5, 2), 10u);
    EXPECT_EQ(gcd(12, 8), 4u);
}

TEST(DomainTest, factorial) {
    EXPECT_EQ(factorial(0), 1u);
    EXPECT_EQ(factorial(1), 1u);
    EXPECT_EQ(factorial(5), 120u);
    EXPECT_EQ(factorial(10), 3628800u);
}

TEST(DomainTest, nchoosek_edge_cases) {
    EXPECT_EQ(nchoosek(5, 0), 1u);
    EXPECT_EQ(nchoosek(5, 5), 1u);
    EXPECT_EQ(nchoosek(10, 3), 120u);
}

TEST(DomainTest, gcd_edge_cases) {
    EXPECT_EQ(gcd(0, 5), 5u);
    EXPECT_EQ(gcd(7, 7), 7u);
    EXPECT_EQ(gcd(48, 18), 6u);
}

TEST(DomainTest, graph_num_edges) {
    Graph g;
    g.n = 3;
    g.adj = {{1, 2}, {0, 2}, {0, 1}};
    EXPECT_EQ(graph_num_edges(g), 3u);
}

TEST(DomainTest, graph_empty) {
    Graph g;
    g.n = 5;
    g.adj.resize(5);
    EXPECT_EQ(graph_num_edges(g), 0u);
}
