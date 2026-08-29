#include <gtest/gtest.h>
#include "ms/poly/poly.hpp"
#include <cmath>

using namespace ms;

// Helpers: polynomial as {a0, a1, a2, ...} where p(x) = a0 + a1*x + a2*x^2 + ...

// ---------------------------------------------------------------------------
// poly_eval
// ---------------------------------------------------------------------------

TEST(PolyExtTest, eval_constant_polynomial) {
    // p(x) = 5
    const std::vector<double> p = {5.0};
    EXPECT_NEAR(poly_eval(p, 0.0)[0], 5.0, 1e-12);
    EXPECT_NEAR(poly_eval(p, 100.0)[0], 5.0, 1e-12);
}

TEST(PolyExtTest, eval_linear_polynomial) {
    // p(x) = 2 + 3x; at x=4: 2+12=14
    const std::vector<double> p = {2.0, 3.0};
    EXPECT_NEAR(poly_eval(p, 4.0)[0], 14.0, 1e-12);
    EXPECT_NEAR(poly_eval(p, 0.0)[0], 2.0, 1e-12);
    EXPECT_NEAR(poly_eval(p, -1.0)[0], -1.0, 1e-12);
}

TEST(PolyExtTest, eval_quadratic_at_zero) {
    // p(x) = 1 + 2x + 3x^2; at x=0 => 1
    const std::vector<double> p = {1.0, 2.0, 3.0};
    EXPECT_NEAR(poly_eval(p, 0.0)[0], 1.0, 1e-12);
}

TEST(PolyExtTest, eval_quadratic_at_one) {
    // p(x) = 1 + 2x + 3x^2; at x=1 => 1+2+3=6
    const std::vector<double> p = {1.0, 2.0, 3.0};
    EXPECT_NEAR(poly_eval(p, 1.0)[0], 6.0, 1e-12);
}

TEST(PolyExtTest, eval_quadratic_at_negative) {
    // p(x) = 1 + 2x + 3x^2; at x=-1 => 1-2+3=2
    const std::vector<double> p = {1.0, 2.0, 3.0};
    EXPECT_NEAR(poly_eval(p, -1.0)[0], 2.0, 1e-12);
}

TEST(PolyExtTest, eval_cubic_known_value) {
    // p(x) = x^3 (coefficients {0,0,0,1}); at x=3 => 27
    const std::vector<double> p = {0.0, 0.0, 0.0, 1.0};
    EXPECT_NEAR(poly_eval(p, 3.0)[0], 27.0, 1e-12);
}

TEST(PolyExtTest, eval_horner_consistency) {
    // p(x) = 1 + x + x^2 + x^3 + x^4; at x=2 => 1+2+4+8+16=31
    const std::vector<double> p = {1.0, 1.0, 1.0, 1.0, 1.0};
    EXPECT_NEAR(poly_eval(p, 2.0)[0], 31.0, 1e-12);
}

// ---------------------------------------------------------------------------
// poly_deriv
// ---------------------------------------------------------------------------

TEST(PolyExtTest, deriv_constant_is_zero) {
    // d/dx 5 = 0
    const std::vector<double> p = {5.0};
    const auto dp = poly_deriv(p);
    EXPECT_NEAR(poly_eval(dp, 0.0)[0], 0.0, 1e-12);
    EXPECT_NEAR(poly_eval(dp, 1.0)[0], 0.0, 1e-12);
}

TEST(PolyExtTest, deriv_linear_is_constant) {
    // d/dx (a + bx) = b
    const std::vector<double> p = {3.0, 7.0};
    const auto dp = poly_deriv(p);
    EXPECT_NEAR(poly_eval(dp, 0.0)[0], 7.0, 1e-12);
    EXPECT_NEAR(poly_eval(dp, 10.0)[0], 7.0, 1e-12);
}

TEST(PolyExtTest, deriv_quadratic) {
    // d/dx (1 + 2x + 3x^2) = 2 + 6x; at x=2 => 2+12=14
    const std::vector<double> p = {1.0, 2.0, 3.0};
    const auto dp = poly_deriv(p);
    EXPECT_NEAR(poly_eval(dp, 2.0)[0], 14.0, 1e-12);
    EXPECT_NEAR(poly_eval(dp, 0.0)[0], 2.0, 1e-12);
}

TEST(PolyExtTest, deriv_cubic) {
    // d/dx x^3 = 3x^2; at x=3 => 27
    const std::vector<double> p = {0.0, 0.0, 0.0, 1.0};
    const auto dp = poly_deriv(p);
    EXPECT_NEAR(poly_eval(dp, 3.0)[0], 27.0, 1e-12);
}

TEST(PolyExtTest, deriv_degree_reduction) {
    // d/dx p(x) has degree one less than p(x)
    const std::vector<double> p = {1.0, 2.0, 3.0, 4.0};
    const auto dp = poly_deriv(p);
    // dp should have size = p.size()-1 = 3
    EXPECT_EQ(dp.size(), 3u);
}

// ---------------------------------------------------------------------------
// poly_add
// ---------------------------------------------------------------------------

TEST(PolyExtTest, add_same_degree) {
    // (1+2x) + (3+4x) = 4+6x
    const std::vector<double> a = {1.0, 2.0};
    const std::vector<double> b = {3.0, 4.0};
    const auto c = poly_add(a, b);
    EXPECT_NEAR(poly_eval(c, 0.0)[0], 4.0, 1e-12);
    EXPECT_NEAR(poly_eval(c, 1.0)[0], 10.0, 1e-12);
}

TEST(PolyExtTest, add_zeros) {
    // p + 0 = p
    const std::vector<double> p = {1.0, 2.0, 3.0};
    const std::vector<double> z = {0.0, 0.0, 0.0};
    const auto result = poly_add(p, z);
    EXPECT_NEAR(poly_eval(result, 2.0)[0], poly_eval(p, 2.0)[0], 1e-12);
}

TEST(PolyExtTest, add_commutativity) {
    // a+b = b+a
    const std::vector<double> a = {1.0, 3.0};
    const std::vector<double> b = {2.0, -1.0};
    const auto ab = poly_add(a, b);
    const auto ba = poly_add(b, a);
    for (size_t i = 0; i < (std::min)(ab.size(), ba.size()); ++i)
        EXPECT_NEAR(ab[i], ba[i], 1e-12);
}

// ---------------------------------------------------------------------------
// poly_sub
// ---------------------------------------------------------------------------

TEST(PolyExtTest, sub_self_is_zero) {
    // p - p = 0 polynomial
    const std::vector<double> p = {1.0, 2.0, 3.0};
    const auto result = poly_sub(p, p);
    for (double v : result)
        EXPECT_NEAR(v, 0.0, 1e-12);
}

TEST(PolyExtTest, sub_difference) {
    // (5+3x) - (2+x) = 3+2x; at x=1 => 5
    const std::vector<double> a = {5.0, 3.0};
    const std::vector<double> b = {2.0, 1.0};
    const auto c = poly_sub(a, b);
    EXPECT_NEAR(poly_eval(c, 1.0)[0], 5.0, 1e-12);
}

// ---------------------------------------------------------------------------
// poly_mul
// ---------------------------------------------------------------------------

TEST(PolyExtTest, mul_by_one) {
    // p * 1 = p
    const std::vector<double> p = {2.0, 3.0, 1.0};
    const std::vector<double> one = {1.0};
    const auto result = poly_mul(p, one);
    EXPECT_NEAR(poly_eval(result, 2.0)[0], poly_eval(p, 2.0)[0], 1e-12);
}

TEST(PolyExtTest, mul_difference_of_squares) {
    // (x+1)*(x-1) = x^2-1; at x=3 => 8
    const std::vector<double> a = {1.0, 1.0};   // 1 + x
    const std::vector<double> b = {-1.0, 1.0};  // -1 + x
    const auto c = poly_mul(a, b);
    EXPECT_NEAR(poly_eval(c, 3.0)[0], 8.0, 1e-12);
    EXPECT_NEAR(poly_eval(c, 0.0)[0], -1.0, 1e-12);
}

TEST(PolyExtTest, mul_degree_sum) {
    // deg(p*q) = deg(p) + deg(q)
    const std::vector<double> a = {1.0, 2.0};        // degree 1
    const std::vector<double> b = {3.0, 4.0, 5.0};  // degree 2
    const auto c = poly_mul(a, b);
    EXPECT_EQ(c.size(), 4u);  // degree 3 = size 4
}

TEST(PolyExtTest, mul_commutativity) {
    // a*b = b*a
    const std::vector<double> a = {1.0, 2.0, 3.0};
    const std::vector<double> b = {4.0, 5.0};
    const auto ab = poly_mul(a, b);
    const auto ba = poly_mul(b, a);
    ASSERT_EQ(ab.size(), ba.size());
    for (size_t i = 0; i < ab.size(); ++i)
        EXPECT_NEAR(ab[i], ba[i], 1e-12);
}
