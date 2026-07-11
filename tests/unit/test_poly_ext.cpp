#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include "ms/poly/poly.hpp"

using namespace ms;
using namespace ms::poly;

static double eval_at(const std::vector<double>& c, double x) {
    return poly_eval(c, x)[0];
}

static void expect_poly_near(const std::vector<double>& a,
                              const std::vector<double>& b,
                              double tol = 1e-9) {
    const size_t n = std::max(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        const double ai = i < a.size() ? a[i] : 0.0;
        const double bi = i < b.size() ? b[i] : 0.0;
        EXPECT_NEAR(ai, bi, tol) << "coeff index " << i;
    }
}

TEST(PolyExtTest, eval_constant) {
    // p(x) = 7 -> p(anything) = 7
    std::vector<double> p{7.0};
    EXPECT_DOUBLE_EQ(poly_eval(p, 0.0)[0], 7.0);
    EXPECT_DOUBLE_EQ(poly_eval(p, 100.0)[0], 7.0);
    EXPECT_DOUBLE_EQ(poly_eval(p, -5.0)[0], 7.0);
}

TEST(PolyExtTest, eval_linear) {
    // p(x) = 2 + 3x -> p(4) = 14
    std::vector<double> p{2.0, 3.0};
    EXPECT_NEAR(poly_eval(p, 4.0)[0], 14.0, 1e-12);
    EXPECT_NEAR(poly_eval(p, 0.0)[0], 2.0, 1e-12);
}

TEST(PolyExtTest, eval_quadratic) {
    // p(x) = x^2 - 1 -> p(3) = 8, p(1) = 0, p(-1) = 0
    std::vector<double> p{-1.0, 0.0, 1.0};
    EXPECT_NEAR(poly_eval(p, 3.0)[0], 8.0, 1e-12);
    EXPECT_NEAR(poly_eval(p, 1.0)[0], 0.0, 1e-12);
    EXPECT_NEAR(poly_eval(p, -1.0)[0], 0.0, 1e-12);
}

TEST(PolyExtTest, deriv_quadratic) {
    // p(x) = 1 + 2x + 3x^2 -> p'(x) = 2 + 6x -> p'(2) = 14
    std::vector<double> p{1.0, 2.0, 3.0};
    const auto dp = poly_deriv(p);
    EXPECT_EQ(dp.size(), 2u);
    EXPECT_NEAR(poly_eval(dp, 2.0)[0], 14.0, 1e-12);
}

TEST(PolyExtTest, deriv_constant_is_zero) {
    std::vector<double> p{5.0};
    const auto dp = poly_deriv(p);
    EXPECT_NEAR(poly_eval(dp, 99.0)[0], 0.0, 1e-12);
}

TEST(PolyExtTest, mul_monomials) {
    // x * x = x^2 -> coeffs [0, 0, 1]
    const auto x = std::vector<double>{0.0, 1.0};
    const auto x2 = poly_mul(x, x);
    EXPECT_EQ(x2.size(), 3u);
    EXPECT_NEAR(x2[0], 0.0, 1e-12);
    EXPECT_NEAR(x2[1], 0.0, 1e-12);
    EXPECT_NEAR(x2[2], 1.0, 1e-12);
}

TEST(PolyExtTest, add_and_sub_inverse) {
    const std::vector<double> a{1.0, 2.0, 3.0};
    const std::vector<double> b{4.0, 5.0};
    const auto sum = poly_add(a, b);
    const auto diff = poly_sub(sum, b);
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(diff[i], a[i], 1e-12);
    }
}

TEST(PolyExtTest, eval_negative_coeffs) {
    // p(x) = -x^2 + x -> p(2) = -4 + 2 = -2
    std::vector<double> p{0.0, 1.0, -1.0};
    EXPECT_NEAR(poly_eval(p, 2.0)[0], -2.0, 1e-12);
}

// ---------------------------------------------------------------------------
// poly_pow
// ---------------------------------------------------------------------------

TEST(PolyExtTest, pow_zero_exponent) {
    const std::vector<double> p{1.0, 2.0, 3.0};
    const auto r = poly_pow(p, 0);
    ASSERT_EQ(r.size(), 1u);
    EXPECT_NEAR(r[0], 1.0, 1e-12);
}

TEST(PolyExtTest, pow_one_exponent) {
    const std::vector<double> p{1.0, 1.0};
    const auto r = poly_pow(p, 1);
    expect_poly_near(r, p);
}

TEST(PolyExtTest, pow_square) {
    // (1 + x)^2 = 1 + 2x + x^2
    const std::vector<double> p{1.0, 1.0};
    const auto r = poly_pow(p, 2);
    expect_poly_near(r, {1.0, 2.0, 1.0});
}

TEST(PolyExtTest, pow_cube_via_squaring) {
    const std::vector<double> p{0.0, 1.0};  // x
    const auto r = poly_pow(p, 3);
    expect_poly_near(r, {0.0, 0.0, 0.0, 1.0});
}

TEST(PolyExtTest, pow_negative_exponent_throws) {
    EXPECT_THROW(poly_pow({1.0, 1.0}, -1), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// poly_monic
// ---------------------------------------------------------------------------

TEST(PolyExtTest, monic_quadratic) {
    const std::vector<double> p{6.0, -5.0, 2.0};  // 2x^2 - 5x + 6
    const auto m = poly_monic(p);
    expect_poly_near(m, {3.0, -2.5, 1.0});
    EXPECT_NEAR(m.back(), 1.0, 1e-12);
}

TEST(PolyExtTest, monic_zero_polynomial) {
    const auto m = poly_monic({0.0});
    ASSERT_EQ(m.size(), 1u);
    EXPECT_NEAR(m[0], 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// poly_reverse
// ---------------------------------------------------------------------------

TEST(PolyExtTest, reverse_quadratic) {
    // p(x) = 1 + 2x + 3x^2 -> x^2 p(1/x) coeffs reversed
    const std::vector<double> p{1.0, 2.0, 3.0};
    const auto r = poly_reverse(p);
    expect_poly_near(r, {3.0, 2.0, 1.0});
}

TEST(PolyExtTest, reverse_twice_is_original) {
    const std::vector<double> p{4.0, -1.0, 2.0, 5.0};
    expect_poly_near(poly_reverse(poly_reverse(p)), p);
}

// ---------------------------------------------------------------------------
// poly_shift — primary check: x^2 shifted by 1 -> (x-1)^2
// ---------------------------------------------------------------------------

TEST(PolyExtTest, shift_x_squared_by_one) {
    const std::vector<double> x2{0.0, 0.0, 1.0};
    const auto s = poly_shift(x2, 1.0);
    expect_poly_near(s, {1.0, -2.0, 1.0});
    EXPECT_NEAR(eval_at(s, 0.0), eval_at(x2, -1.0), 1e-10);
}

TEST(PolyExtTest, shift_identity_at_zero) {
    const std::vector<double> p{1.0, 2.0, 3.0};
    expect_poly_near(poly_shift(p, 0.0), p);
}

TEST(PolyExtTest, shift_linear) {
    // x + 3 at a=2 -> (x-2)+3 = x+1
    const auto s = poly_shift({3.0, 1.0}, 2.0);
    expect_poly_near(s, {1.0, 1.0});
}

// ---------------------------------------------------------------------------
// poly_scale
// ---------------------------------------------------------------------------

TEST(PolyExtTest, scale_x_squared) {
    // p(x)=x^2, p(2x)=4x^2
    const auto s = poly_scale({0.0, 0.0, 1.0}, 2.0);
    expect_poly_near(s, {0.0, 0.0, 4.0});
}

TEST(PolyExtTest, scale_eval_relation) {
    const std::vector<double> p{1.0, -2.0, 3.0};
    const double a = 2.5;
    const auto s = poly_scale(p, a);
    EXPECT_NEAR(eval_at(s, 1.0), eval_at(p, a), 1e-10);
}

// ---------------------------------------------------------------------------
// poly_lcm
// ---------------------------------------------------------------------------

TEST(PolyExtTest, lcm_identity_gcd_times_lcm) {
    const std::vector<double> a{-1.0, 0.0, 1.0};  // x^2 - 1
    const std::vector<double> b{-1.0, 1.0};       // x - 1
    const auto g = poly_gcd(a, b);
    const auto l = poly_lcm(a, b);
    const auto prod = poly_mul(a, b);
    const auto check = poly_mul(g, l);
    EXPECT_NEAR(eval_at(check, 2.0), eval_at(prod, 2.0), 1e-8);
    EXPECT_NEAR(l.back(), 1.0, 1e-8);
}

TEST(PolyExtTest, lcm_coprime_is_product) {
    const std::vector<double> a{-1.0, 1.0};  // x - 1
    const std::vector<double> b{1.0, 1.0};   // x + 1
    const auto l = poly_lcm(a, b);
    const auto prod = poly_mul(a, b);
    EXPECT_NEAR(eval_at(l, 3.0), eval_at(prod, 3.0), 1e-8);
}

// ---------------------------------------------------------------------------
// poly_sylvester — Wikipedia-style worked example
// ---------------------------------------------------------------------------

TEST(PolyExtTest, sylvester_worked_example) {
    // p(x)=x^2+2x+3, q(x)=x+5
    const std::vector<double> p{3.0, 2.0, 1.0};
    const std::vector<double> q{5.0, 1.0};
    const auto S = poly_sylvester(p, q);
    ASSERT_EQ(S.rows(), 3u);
    ASSERT_EQ(S.cols(), 3u);
    EXPECT_NEAR(S(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(S(0, 1), 2.0, 1e-12);
    EXPECT_NEAR(S(0, 2), 3.0, 1e-12);
    EXPECT_NEAR(S(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(S(1, 1), 5.0, 1e-12);
    EXPECT_NEAR(S(1, 2), 0.0, 1e-12);
    EXPECT_NEAR(S(2, 0), 0.0, 1e-12);
    EXPECT_NEAR(S(2, 1), 1.0, 1e-12);
    EXPECT_NEAR(S(2, 2), 5.0, 1e-12);
}

// ---------------------------------------------------------------------------
// poly_resultant
// ---------------------------------------------------------------------------

TEST(PolyExtTest, resultant_worked_example) {
    const std::vector<double> p{3.0, 2.0, 1.0};
    const std::vector<double> q{5.0, 1.0};
    EXPECT_NEAR(poly_resultant(p, q), 18.0, 1e-8);
}

TEST(PolyExtTest, resultant_zero_when_common_root) {
    const std::vector<double> p{6.0, -5.0, 1.0};   // (x-2)(x-3)
    const std::vector<double> q{10.0, -7.0, 1.0};  // (x-2)(x-5)
    EXPECT_NEAR(poly_resultant(p, q), 0.0, 1e-8);
}

TEST(PolyExtTest, resultant_nonzero_when_coprime) {
    const std::vector<double> p{6.0, -5.0, 1.0};   // (x-2)(x-3)
    const std::vector<double> q{1.0, 0.0, 1.0};  // x^2 + 1, no common real roots
    const double res = poly_resultant(p, q);
    EXPECT_GT(std::abs(res), 1.0);
}

// ---------------------------------------------------------------------------
// poly_discriminant
// ---------------------------------------------------------------------------

TEST(PolyExtTest, discriminant_quadratic_formula) {
    // ax^2+bx+c: disc = b^2 - 4ac
    const double a = 2.0, b = -5.0, c = 3.0;
    const std::vector<double> p{c, b, a};
    const double expected = b * b - 4.0 * a * c;
    EXPECT_NEAR(poly_discriminant(p), expected, 1e-8);
}

TEST(PolyExtTest, discriminant_repeated_root_is_zero) {
    // (x-2)^2 = x^2 - 4x + 4
    const std::vector<double> p{4.0, -4.0, 1.0};
    EXPECT_NEAR(poly_discriminant(p), 0.0, 1e-8);
}

TEST(PolyExtTest, discriminant_x_squared_minus_one) {
    const std::vector<double> p{-1.0, 0.0, 1.0};
    EXPECT_NEAR(poly_discriminant(p), 4.0, 1e-8);
}

// ---------------------------------------------------------------------------
// poly_squarefree
// ---------------------------------------------------------------------------

TEST(PolyExtTest, gcd_repeated_root_factor) {
    const std::vector<double> p{-12.0, 16.0, -7.0, 1.0};
    const auto g = poly_gcd(p, poly_deriv(p));
    ASSERT_GE(g.size(), 2u);
    EXPECT_NEAR(eval_at(g, 2.0), 0.0, 1e-6);
}

TEST(PolyExtTest, squarefree_removes_multiplicity) {
    // (x-2)^2 (x-3) = x^3 - 7x^2 + 16x - 12
    const std::vector<double> p{-12.0, 16.0, -7.0, 1.0};
    const auto sf = poly_squarefree(p);
    const auto roots = poly_roots(sf);
    ASSERT_EQ(roots.size(), 2u);
    std::vector<double> real_roots;
    for (const auto& r : roots) {
        EXPECT_NEAR(r.imag(), 0.0, 1e-5);
        real_roots.push_back(r.real());
    }
    std::sort(real_roots.begin(), real_roots.end());
    EXPECT_NEAR(real_roots[0], 2.0, 1e-4);
    EXPECT_NEAR(real_roots[1], 3.0, 1e-4);
}

TEST(PolyExtTest, squarefree_already_squarefree) {
    const std::vector<double> p{6.0, -5.0, 1.0};
    const auto sf = poly_squarefree(p);
    expect_poly_near(sf, p);
}

// ---------------------------------------------------------------------------
// bernstein
// ---------------------------------------------------------------------------

TEST(PolyExtTest, bernstein_small_case) {
    // B_{2,1}(0.5) = C(2,1)*0.5*0.5 = 0.5
    EXPECT_NEAR(bernstein(2, 1, 0.5), 0.5, 1e-12);
}

TEST(PolyExtTest, bernstein_endpoints) {
    EXPECT_NEAR(bernstein(3, 0, 0.0), 1.0, 1e-12);
    EXPECT_NEAR(bernstein(3, 3, 1.0), 1.0, 1e-12);
    EXPECT_NEAR(bernstein(3, 1, 0.0), 0.0, 1e-12);
}

TEST(PolyExtTest, bernstein_partition_of_unity) {
    for (double x : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        for (int n : {1, 2, 4, 6}) {
            double sum = 0.0;
            for (int i = 0; i <= n; ++i) {
                sum += bernstein(n, i, x);
            }
            EXPECT_NEAR(sum, 1.0, 1e-12) << "n=" << n << " x=" << x;
        }
    }
}
