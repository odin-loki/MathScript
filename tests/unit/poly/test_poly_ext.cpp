#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <span>
#include <utility>
#include "ms/poly/poly.hpp"

using namespace ms;
using namespace ms::poly;

static double eval_at(const std::vector<double>& c, double x) {
    return poly_eval(c, x)[0];
}

// Multiset-style comparison of found roots against expected {root_value -> multiplicity},
// independent of the order poly_rational_roots happens to return them in.
static void expect_roots_multiset(
    const std::vector<std::pair<int64_t, int64_t>>& found,
    const std::map<std::pair<int64_t, int64_t>, int>& expected) {
    ASSERT_EQ(found.size(),
              std::accumulate(expected.begin(), expected.end(), size_t{0},
                               [](size_t acc, const auto& kv) {
                                   return acc + static_cast<size_t>(kv.second);
                               }));
    std::map<std::pair<int64_t, int64_t>, int> counts;
    for (const auto& r : found) counts[r]++;
    for (const auto& [root, mult] : expected) {
        EXPECT_EQ(counts[root], mult)
            << "root " << root.first << "/" << root.second;
    }
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
    ASSERT_TRUE(r.has_value());
    const auto& x = *r;
    ASSERT_EQ(x.size(), 1u);
    EXPECT_NEAR(x[0], 1.0, 1e-12);
}

TEST(PolyExtTest, pow_one_exponent) {
    const std::vector<double> p{1.0, 1.0};
    const auto r = poly_pow(p, 1);
    ASSERT_TRUE(r.has_value());
    expect_poly_near(*r, p);
}

TEST(PolyExtTest, pow_square) {
    // (1 + x)^2 = 1 + 2x + x^2
    const std::vector<double> p{1.0, 1.0};
    const auto r = poly_pow(p, 2);
    ASSERT_TRUE(r.has_value());
    expect_poly_near(*r, {1.0, 2.0, 1.0});
}

TEST(PolyExtTest, pow_cube_via_squaring) {
    const std::vector<double> p{0.0, 1.0};  // x
    const auto r = poly_pow(p, 3);
    ASSERT_TRUE(r.has_value());
    expect_poly_near(*r, {0.0, 0.0, 0.0, 1.0});
}

TEST(PolyExtTest, pow_negative_exponent_throws) {
    const auto r = poly_pow({1.0, 1.0}, -1);
    ASSERT_FALSE(r.has_value());
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

// ---------------------------------------------------------------------------
// interp_newton
// ---------------------------------------------------------------------------

TEST(PolyExtTest, newton_matches_lagrange_three_points) {
    const std::vector<double> xs{0.0, 1.0, 2.0};
    const std::vector<double> ys{1.0, 3.0, 7.0};
    const auto pn = interp_newton(xs, ys);
    const auto pl = poly_lagrange(xs, ys);
    for (double x : {-1.0, 0.5, 1.5, 3.0, 10.0}) {
        EXPECT_NEAR(eval_at(pn, x), eval_at(pl, x), 1e-9) << "x=" << x;
    }
}

TEST(PolyExtTest, newton_matches_lagrange_four_points) {
    const std::vector<double> xs{-1.0, 0.0, 1.0, 2.0};
    const std::vector<double> ys{2.0, 1.0, 0.0, 5.0};
    const auto pn = interp_newton(xs, ys);
    const auto pl = poly_lagrange(xs, ys);
    for (double x : {-2.0, -0.5, 0.75, 1.5, 4.0}) {
        EXPECT_NEAR(eval_at(pn, x), eval_at(pl, x), 1e-9) << "x=" << x;
    }
}

TEST(PolyExtTest, newton_matches_lagrange_five_points) {
    const std::vector<double> xs{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> ys{1.0, 2.0, 0.0, -3.0, 5.0};
    const auto pn = interp_newton(xs, ys);
    const auto pl = poly_lagrange(xs, ys);
    for (double x : {-1.0, 0.5, 2.5, 3.5, 6.0}) {
        EXPECT_NEAR(eval_at(pn, x), eval_at(pl, x), 1e-8) << "x=" << x;
    }
}

TEST(PolyExtTest, newton_reproduces_input_points_three) {
    const std::vector<double> xs{0.0, 1.0, 2.0};
    const std::vector<double> ys{1.0, 3.0, 7.0};
    const auto p = interp_newton(xs, ys);
    for (size_t i = 0; i < xs.size(); ++i) {
        EXPECT_NEAR(eval_at(p, xs[i]), ys[i], 1e-9) << "i=" << i;
    }
}

TEST(PolyExtTest, newton_reproduces_input_points_four) {
    const std::vector<double> xs{-2.0, -0.5, 1.0, 3.0};
    const std::vector<double> ys{4.0, -1.0, 2.0, 9.0};
    const auto p = interp_newton(xs, ys);
    for (size_t i = 0; i < xs.size(); ++i) {
        EXPECT_NEAR(eval_at(p, xs[i]), ys[i], 1e-8) << "i=" << i;
    }
}

TEST(PolyExtTest, newton_reproduces_input_points_five) {
    const std::vector<double> xs{-1.0, 0.0, 1.0, 2.5, 4.0};
    const std::vector<double> ys{3.0, -2.0, 0.0, 5.0, 1.0};
    const auto p = interp_newton(xs, ys);
    for (size_t i = 0; i < xs.size(); ++i) {
        EXPECT_NEAR(eval_at(p, xs[i]), ys[i], 1e-7) << "i=" << i;
    }
}

TEST(PolyExtTest, newton_single_point_is_constant) {
    const auto p = interp_newton({3.0}, {9.0});
    ASSERT_EQ(p.size(), 1u);
    EXPECT_NEAR(p[0], 9.0, 1e-12);
    EXPECT_NEAR(eval_at(p, 100.0), 9.0, 1e-12);
}

TEST(PolyExtTest, newton_two_points_is_line) {
    // through (0,1) and (2,5): slope 2, p(x) = 1 + 2x
    const auto p = interp_newton({0.0, 2.0}, {1.0, 5.0});
    expect_poly_near(p, {1.0, 2.0});
}

TEST(PolyExtTest, newton_empty_input_returns_empty) {
    EXPECT_TRUE(interp_newton({}, {}).empty());
}

TEST(PolyExtTest, newton_size_mismatch_returns_empty) {
    EXPECT_TRUE(interp_newton({0.0, 1.0}, {1.0}).empty());
}

TEST(PolyExtTest, newton_duplicate_xs_returns_empty) {
    EXPECT_TRUE(interp_newton({1.0, 1.0, 2.0}, {1.0, 2.0, 3.0}).empty());
}

// ---------------------------------------------------------------------------
// interp_hermite
// ---------------------------------------------------------------------------

TEST(PolyExtTest, hermite_tangent_line_single_point) {
    // p(x) = 5 + 3*(x-2) = -1 + 3x
    const auto p = interp_hermite({2.0}, {5.0}, {3.0});
    expect_poly_near(p, {-1.0, 3.0});
    EXPECT_NEAR(eval_at(p, 2.0), 5.0, 1e-12);
    EXPECT_NEAR(eval_at(poly_deriv(p), 2.0), 3.0, 1e-12);
}

TEST(PolyExtTest, hermite_reproduces_values_and_derivatives_two_points_cubic) {
    // f(x) = x^3, f'(x) = 3x^2, sampled at x = 1, 2
    const std::vector<double> xs{1.0, 2.0};
    const std::vector<double> ys{1.0, 8.0};
    const std::vector<double> dys{3.0, 12.0};
    const auto p = interp_hermite(xs, ys, dys);
    const auto dp = poly_deriv(p);
    for (size_t i = 0; i < xs.size(); ++i) {
        EXPECT_NEAR(eval_at(p, xs[i]), ys[i], 1e-8) << "i=" << i;
        EXPECT_NEAR(eval_at(dp, xs[i]), dys[i], 1e-7) << "i=" << i;
    }
}

TEST(PolyExtTest, hermite_reproduces_values_and_derivatives_three_points_cubic) {
    // f(x) = x^3, f'(x) = 3x^2, sampled at x = -1, 0, 2
    const std::vector<double> xs{-1.0, 0.0, 2.0};
    const std::vector<double> ys{-1.0, 0.0, 8.0};
    const std::vector<double> dys{3.0, 0.0, 12.0};
    const auto p = interp_hermite(xs, ys, dys);
    const auto dp = poly_deriv(p);
    for (size_t i = 0; i < xs.size(); ++i) {
        EXPECT_NEAR(eval_at(p, xs[i]), ys[i], 1e-7) << "i=" << i;
        EXPECT_NEAR(eval_at(dp, xs[i]), dys[i], 1e-6) << "i=" << i;
    }
}

TEST(PolyExtTest, hermite_reproduces_values_and_derivatives_sine) {
    // f(x) = sin(x), f'(x) = cos(x), sampled at a few points
    const std::vector<double> xs{0.0, 0.5, 1.0, 1.5};
    std::vector<double> ys, dys;
    for (double x : xs) {
        ys.push_back(std::sin(x));
        dys.push_back(std::cos(x));
    }
    const auto p = interp_hermite(xs, ys, dys);
    const auto dp = poly_deriv(p);
    for (size_t i = 0; i < xs.size(); ++i) {
        EXPECT_NEAR(eval_at(p, xs[i]), ys[i], 1e-6) << "i=" << i;
        EXPECT_NEAR(eval_at(dp, xs[i]), dys[i], 1e-5) << "i=" << i;
    }
}

TEST(PolyExtTest, hermite_degree_bound_two_points) {
    // n=2 points -> degree < 4, so at most 4 coefficients
    const auto p = interp_hermite({0.0, 1.0}, {0.0, 1.0}, {0.0, 3.0});
    EXPECT_LE(p.size(), 4u);
}

TEST(PolyExtTest, hermite_empty_input_returns_empty) {
    EXPECT_TRUE(interp_hermite({}, {}, {}).empty());
}

TEST(PolyExtTest, hermite_size_mismatch_returns_empty) {
    EXPECT_TRUE(interp_hermite({0.0, 1.0}, {0.0, 1.0}, {0.0}).empty());
}

TEST(PolyExtTest, hermite_duplicate_xs_returns_empty) {
    EXPECT_TRUE(interp_hermite({1.0, 1.0}, {2.0, 3.0}, {1.0, 1.0}).empty());
}

// ---------------------------------------------------------------------------
// poly_rational_roots / poly_factor_rational
// ---------------------------------------------------------------------------

static std::vector<double> reconstruct_from_factorization(const RationalFactorization& f) {
    std::vector<double> poly = f.remainder;
    for (const auto& [num, den] : f.linear_roots) {
        const std::vector<double> factor{static_cast<double>(-num), static_cast<double>(den)};
        poly = poly_mul(poly, factor);
    }
    return poly;
}

// Cross-verify: every reported root, when evaluated against the ORIGINAL
// polynomial via poly_eval, is (near) zero.
static void expect_all_roots_evaluate_to_zero(
    const std::vector<double>& original,
    const std::vector<std::pair<int64_t, int64_t>>& roots) {
    for (const auto& [num, den] : roots) {
        const double x = static_cast<double>(num) / static_cast<double>(den);
        EXPECT_NEAR(eval_at(original, x), 0.0, 1e-6)
            << "root " << num << "/" << den;
    }
}

TEST(PolyExtTest, rational_roots_three_distinct_integer_roots) {
    // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
    const std::vector<double> p{-6.0, 11.0, -6.0, 1.0};
    const auto res = poly_rational_roots(p);
    ASSERT_TRUE(res.has_value());
    expect_roots_multiset(*res, {{{1, 1}, 1}, {{2, 1}, 1}, {{3, 1}, 1}});
    expect_all_roots_evaluate_to_zero(p, *res);
}

TEST(PolyExtTest, factor_rational_three_distinct_integer_roots_fully_factors) {
    const std::vector<double> p{-6.0, 11.0, -6.0, 1.0};
    const auto res = poly_factor_rational(p);
    ASSERT_TRUE(res.has_value());
    expect_roots_multiset(res->linear_roots, {{{1, 1}, 1}, {{2, 1}, 1}, {{3, 1}, 1}});
    ASSERT_EQ(res->remainder.size(), 1u);
    EXPECT_NEAR(res->remainder[0], 1.0, 1e-9);
    expect_poly_near(reconstruct_from_factorization(*res), p, 1e-6);
}

TEST(PolyExtTest, rational_roots_repeated_root_multiplicity_two) {
    // (x-2)^2 (x+1) = x^3 - 3x^2 + 0x + 4
    const std::vector<double> p{4.0, 0.0, -3.0, 1.0};
    const auto res = poly_rational_roots(p);
    ASSERT_TRUE(res.has_value());
    expect_roots_multiset(*res, {{{2, 1}, 2}, {{-1, 1}, 1}});
    expect_all_roots_evaluate_to_zero(p, *res);
}

TEST(PolyExtTest, factor_rational_repeated_root_reconstructs_original) {
    const std::vector<double> p{4.0, 0.0, -3.0, 1.0};
    const auto res = poly_factor_rational(p);
    ASSERT_TRUE(res.has_value());
    expect_roots_multiset(res->linear_roots, {{{2, 1}, 2}, {{-1, 1}, 1}});
    ASSERT_EQ(res->remainder.size(), 1u);
    expect_poly_near(reconstruct_from_factorization(*res), p, 1e-6);
}

TEST(PolyExtTest, rational_roots_non_integer_rational_roots) {
    // (2x-1)(3x+2) = 6x^2 + x - 2
    const std::vector<double> p{-2.0, 1.0, 6.0};
    const auto res = poly_rational_roots(p);
    ASSERT_TRUE(res.has_value());
    expect_roots_multiset(*res, {{{1, 2}, 1}, {{-2, 3}, 1}});
    expect_all_roots_evaluate_to_zero(p, *res);
}

TEST(PolyExtTest, factor_rational_non_integer_rational_roots_reconstructs) {
    const std::vector<double> p{-2.0, 1.0, 6.0};
    const auto res = poly_factor_rational(p);
    ASSERT_TRUE(res.has_value());
    expect_roots_multiset(res->linear_roots, {{{1, 2}, 1}, {{-2, 3}, 1}});
    expect_poly_near(reconstruct_from_factorization(*res), p, 1e-6);
}

TEST(PolyExtTest, rational_roots_no_real_roots_x_squared_plus_one) {
    const std::vector<double> p{1.0, 0.0, 1.0};  // x^2 + 1
    const auto res = poly_rational_roots(p);
    ASSERT_TRUE(res.has_value());
    EXPECT_TRUE(res->empty());
}

TEST(PolyExtTest, factor_rational_no_real_roots_remainder_is_original) {
    const std::vector<double> p{1.0, 0.0, 1.0};  // x^2 + 1
    const auto res = poly_factor_rational(p);
    ASSERT_TRUE(res.has_value());
    EXPECT_TRUE(res->linear_roots.empty());
    expect_poly_near(res->remainder, p, 1e-9);
}

TEST(PolyExtTest, rational_roots_irrational_roots_x_squared_minus_two) {
    const std::vector<double> p{-2.0, 0.0, 1.0};  // x^2 - 2, roots +/- sqrt(2)
    const auto res = poly_rational_roots(p);
    ASSERT_TRUE(res.has_value());
    EXPECT_TRUE(res->empty());
}

TEST(PolyExtTest, factor_rational_irrational_roots_remainder_is_original) {
    const std::vector<double> p{-2.0, 0.0, 1.0};
    const auto res = poly_factor_rational(p);
    ASSERT_TRUE(res.has_value());
    EXPECT_TRUE(res->linear_roots.empty());
    expect_poly_near(res->remainder, p, 1e-9);
}

TEST(PolyExtTest, rational_roots_mixed_rational_and_irrational) {
    // (x-1)(x^2-2) = x^3 - x^2 - 2x + 2
    const std::vector<double> p{2.0, -2.0, -1.0, 1.0};
    const auto res = poly_rational_roots(p);
    ASSERT_TRUE(res.has_value());
    expect_roots_multiset(*res, {{{1, 1}, 1}});
    expect_all_roots_evaluate_to_zero(p, *res);
}

TEST(PolyExtTest, factor_rational_mixed_rational_and_irrational_remainder) {
    // (x-1)(x^2-2) = x^3 - x^2 - 2x + 2; remainder should be x^2 - 2 (up to scale)
    const std::vector<double> p{2.0, -2.0, -1.0, 1.0};
    const auto res = poly_factor_rational(p);
    ASSERT_TRUE(res.has_value());
    expect_roots_multiset(res->linear_roots, {{{1, 1}, 1}});
    expect_poly_near(res->remainder, {-2.0, 0.0, 1.0}, 1e-6);
    expect_poly_near(reconstruct_from_factorization(*res), p, 1e-6);
}

TEST(PolyExtTest, rational_roots_zero_polynomial_is_error) {
    const auto res1 = poly_rational_roots({0.0});
    EXPECT_FALSE(res1.has_value());
    const auto res2 = poly_rational_roots({});
    EXPECT_FALSE(res2.has_value());
}

TEST(PolyExtTest, factor_rational_zero_polynomial_is_error) {
    const auto res = poly_factor_rational({0.0});
    EXPECT_FALSE(res.has_value());
}

TEST(PolyExtTest, rational_roots_constant_polynomial_has_no_roots) {
    const auto res = poly_rational_roots({5.0});
    ASSERT_TRUE(res.has_value());
    EXPECT_TRUE(res->empty());
}

TEST(PolyExtTest, factor_rational_constant_polynomial_remainder_is_itself) {
    const auto res = poly_factor_rational({5.0});
    ASSERT_TRUE(res.has_value());
    EXPECT_TRUE(res->linear_roots.empty());
    ASSERT_EQ(res->remainder.size(), 1u);
    EXPECT_NEAR(res->remainder[0], 5.0, 1e-9);
}

TEST(PolyExtTest, rational_roots_degree_one_polynomial) {
    // 2x - 3 -> root 3/2
    const std::vector<double> p{-3.0, 2.0};
    const auto res = poly_rational_roots(p);
    ASSERT_TRUE(res.has_value());
    expect_roots_multiset(*res, {{{3, 2}, 1}});
    expect_all_roots_evaluate_to_zero(p, *res);
}

TEST(PolyExtTest, rational_roots_zero_is_a_root) {
    // x^2 - x = x(x-1)
    const std::vector<double> p{0.0, -1.0, 1.0};
    const auto res = poly_rational_roots(p);
    ASSERT_TRUE(res.has_value());
    expect_roots_multiset(*res, {{{0, 1}, 1}, {{1, 1}, 1}});
    expect_all_roots_evaluate_to_zero(p, *res);
}

TEST(PolyExtTest, rational_roots_repeated_zero_root) {
    // x^3 - x^2 = x^2(x-1)
    const std::vector<double> p{0.0, 0.0, -1.0, 1.0};
    const auto res = poly_rational_roots(p);
    ASSERT_TRUE(res.has_value());
    expect_roots_multiset(*res, {{{0, 1}, 2}, {{1, 1}, 1}});
    expect_all_roots_evaluate_to_zero(p, *res);
}

TEST(PolyExtTest, rational_roots_non_integer_coefficients_is_error) {
    // 0.5x^2 + 1 is not within tolerance of integer coefficients
    const auto res = poly_rational_roots({1.0, 0.0, 0.5});
    EXPECT_FALSE(res.has_value());
}

TEST(PolyExtTest, rational_roots_large_but_within_limit_coefficients) {
    // (1000000x - 1)(x - 1) = 1000000x^2 - 1000001x + 1
    const std::vector<double> p{1.0, -1000001.0, 1000000.0};
    const auto res = poly_rational_roots(p);
    ASSERT_TRUE(res.has_value());
    expect_roots_multiset(*res, {{{1, 1000000}, 1}, {{1, 1}, 1}});
    expect_all_roots_evaluate_to_zero(p, *res);
}

TEST(PolyExtTest, rational_roots_exceeds_safety_limit_is_error) {
    // Constant term 1e8 exceeds the documented safety limit (1e7).
    const std::vector<double> p{100000000.0, 0.0, 1.0};
    const auto res = poly_rational_roots(p);
    EXPECT_FALSE(res.has_value());
}

TEST(PolyExtTest, factor_rational_negative_leading_coefficient) {
    // -(x-1)(x-2) = -x^2 + 3x - 2
    const std::vector<double> p{-2.0, 3.0, -1.0};
    const auto res = poly_factor_rational(p);
    ASSERT_TRUE(res.has_value());
    expect_roots_multiset(res->linear_roots, {{{1, 1}, 1}, {{2, 1}, 1}});
    expect_poly_near(reconstruct_from_factorization(*res), p, 1e-6);
}

TEST(PolyExtTest, rational_roots_near_integer_coefficients_within_tolerance) {
    // Same as (x-1)(x-2)(x-3) but with tiny floating-point noise, as would
    // arise from expanding via poly_mul in double precision.
    const auto expanded = poly_mul(poly_mul({-1.0, 1.0}, {-2.0, 1.0}), {-3.0, 1.0});
    std::vector<double> noisy = expanded;
    for (auto& c : noisy) c += 1e-10;
    const auto res = poly_rational_roots(noisy);
    ASSERT_TRUE(res.has_value());
    expect_roots_multiset(*res, {{{1, 1}, 1}, {{2, 1}, 1}, {{3, 1}, 1}});
}

// ---------------------------------------------------------------------------
// poly_factor (real factorization via poly_roots)
// ---------------------------------------------------------------------------

static std::vector<double> reconstruct_from_poly_factors(
    const std::vector<PolyFactor>& factors) {
    std::vector<double> poly = {1.0};
    for (const auto& f : factors) {
        std::vector<double> term = f.coeffs;
        for (int i = 1; i < f.multiplicity; ++i) {
            term = poly_mul(term, f.coeffs);
        }
        poly = poly_mul(poly, term);
    }
    return poly;
}

static double linear_factor_root(const std::vector<double>& c) {
    return -c[0] / c[1];
}

static void expect_linear_roots_near(const std::vector<PolyFactor>& factors,
                                      const std::vector<double>& expected_roots,
                                      double tol = 1e-4) {
    std::vector<double> roots;
    for (const auto& f : factors) {
        ASSERT_EQ(f.coeffs.size(), 2u);
        for (int i = 0; i < f.multiplicity; ++i) {
            roots.push_back(linear_factor_root(f.coeffs));
        }
    }
    ASSERT_EQ(roots.size(), expected_roots.size());
    std::sort(roots.begin(), roots.end());
    auto sorted_expected = expected_roots;
    std::sort(sorted_expected.begin(), sorted_expected.end());
    for (size_t i = 0; i < roots.size(); ++i) {
        EXPECT_NEAR(roots[i], sorted_expected[i], tol) << "root index " << i;
    }
}

TEST(PolyExtTest, poly_factor_three_distinct_real_roots) {
    // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
    const std::vector<double> p{-6.0, 11.0, -6.0, 1.0};
    const auto factors = poly_factor(p);
    ASSERT_EQ(factors.size(), 3u);
    for (const auto& f : factors) {
        EXPECT_EQ(f.multiplicity, 1);
    }
    expect_linear_roots_near(factors, {1.0, 2.0, 3.0});
    expect_poly_near(reconstruct_from_poly_factors(factors), p, 1e-5);
}

TEST(PolyExtTest, poly_factor_constant_polynomial) {
    const auto factors = poly_factor({7.0});
    ASSERT_EQ(factors.size(), 1u);
    EXPECT_EQ(factors[0].multiplicity, 1);
    ASSERT_EQ(factors[0].coeffs.size(), 1u);
    EXPECT_NEAR(factors[0].coeffs[0], 7.0, 1e-12);
}

TEST(PolyExtTest, poly_factor_linear_polynomial) {
    // 2 + 3x returned as a single irreducible linear factor
    const auto factors = poly_factor({2.0, 3.0});
    ASSERT_EQ(factors.size(), 1u);
    EXPECT_EQ(factors[0].multiplicity, 1);
    expect_poly_near(factors[0].coeffs, {2.0, 3.0}, 1e-12);
    expect_poly_near(reconstruct_from_poly_factors(factors), {2.0, 3.0}, 1e-9);
}

TEST(PolyExtTest, poly_factor_zero_polynomial_returns_empty) {
    EXPECT_TRUE(poly_factor({0.0}).empty());
    EXPECT_TRUE(poly_factor({}).empty());
}

TEST(PolyExtTest, poly_factor_quadratic_complex_roots) {
    // x^2 + 1 -> irreducible quadratic {1, 0, 1}
    const std::vector<double> p{1.0, 0.0, 1.0};
    const auto factors = poly_factor(p);
    ASSERT_EQ(factors.size(), 1u);
    EXPECT_EQ(factors[0].multiplicity, 1);
    ASSERT_EQ(factors[0].coeffs.size(), 3u);
    EXPECT_NEAR(factors[0].coeffs[0], 1.0, 1e-9);
    EXPECT_NEAR(factors[0].coeffs[1], 0.0, 1e-9);
    EXPECT_NEAR(factors[0].coeffs[2], 1.0, 1e-9);
    expect_poly_near(reconstruct_from_poly_factors(factors), p, 1e-5);
}

TEST(PolyExtTest, poly_factor_repeated_real_root) {
    // (x-2)^2 = x^2 - 4x + 4
    const std::vector<double> p{4.0, -4.0, 1.0};
    const auto factors = poly_factor(p);
    ASSERT_EQ(factors.size(), 1u);
    EXPECT_EQ(factors[0].multiplicity, 2);
    expect_linear_roots_near(factors, {2.0, 2.0});
    expect_poly_near(reconstruct_from_poly_factors(factors), p, 1e-4);
}

TEST(PolyExtTest, poly_factor_mixed_real_and_quadratic) {
    // (x-1)(x^2+1) = x^3 - x^2 + x - 1
    const std::vector<double> p{-1.0, 1.0, -1.0, 1.0};
    const auto factors = poly_factor(p);
    ASSERT_EQ(factors.size(), 2u);
    int linear = 0;
    int quadratic = 0;
    for (const auto& f : factors) {
        EXPECT_EQ(f.multiplicity, 1);
        if (f.coeffs.size() == 2u) {
            ++linear;
            EXPECT_NEAR(linear_factor_root(f.coeffs), 1.0, 1e-4);
        } else if (f.coeffs.size() == 3u) {
            ++quadratic;
            expect_poly_near(f.coeffs, {1.0, 0.0, 1.0}, 1e-4);
        }
    }
    EXPECT_EQ(linear, 1);
    EXPECT_EQ(quadratic, 1);
    expect_poly_near(reconstruct_from_poly_factors(factors), p, 1e-5);
}

TEST(PolyExtTest, poly_factor_non_monic_leading_coefficient) {
    // 2(x-1)(x-2) = 2x^2 - 6x + 4
    const std::vector<double> p{4.0, -6.0, 2.0};
    const auto factors = poly_factor(p);
    ASSERT_EQ(factors.size(), 3u);
    int constant = 0;
    int linear = 0;
    for (const auto& f : factors) {
        EXPECT_EQ(f.multiplicity, 1);
        if (f.coeffs.size() == 1u) {
            ++constant;
            EXPECT_NEAR(f.coeffs[0], 2.0, 1e-9);
        } else {
            ++linear;
        }
    }
    EXPECT_EQ(constant, 1);
    EXPECT_EQ(linear, 2);
    expect_linear_roots_near(
        std::vector<PolyFactor>(factors.begin() + 1, factors.end()), {1.0, 2.0});
    expect_poly_near(reconstruct_from_poly_factors(factors), p, 1e-5);
}

TEST(PolyExtTest, poly_factor_repeated_and_simple_real_roots) {
    // (x-2)^2 (x+1) = x^3 - 3x^2 + 0x + 4
    const std::vector<double> p{4.0, 0.0, -3.0, 1.0};
    const auto factors = poly_factor(p);
    ASSERT_EQ(factors.size(), 2u);
    int mult_two = 0;
    int mult_one = 0;
    for (const auto& f : factors) {
        if (f.multiplicity == 2) {
            ++mult_two;
            EXPECT_NEAR(linear_factor_root(f.coeffs), 2.0, 1e-4);
        } else if (f.multiplicity == 1) {
            ++mult_one;
            EXPECT_NEAR(linear_factor_root(f.coeffs), -1.0, 1e-4);
        }
    }
    EXPECT_EQ(mult_two, 1);
    EXPECT_EQ(mult_one, 1);
    expect_poly_near(reconstruct_from_poly_factors(factors), p, 1e-4);
}

// ---------------------------------------------------------------------------
// poly_partial_fractions
// ---------------------------------------------------------------------------

TEST(PolyPartialFractions, EmptyDenomReturnsEmpty) {
    const std::vector<double> num{1.0, 2.0};
    const auto empty_den = poly_partial_fractions(num, {});
    EXPECT_TRUE(empty_den.quotient.empty());
    EXPECT_TRUE(empty_den.terms.empty());
    const auto zero_den = poly_partial_fractions(num, {0.0});
    EXPECT_TRUE(zero_den.quotient.empty());
    EXPECT_TRUE(zero_den.terms.empty());
}

TEST(PolyPartialFractions, SimpleRealPole) {
    // 1 / (x - 1)
    const std::vector<double> num{1.0};
    const std::vector<double> den{-1.0, 1.0};
    const auto res = poly_partial_fractions(num, den);
    EXPECT_TRUE(res.quotient.empty());
    ASSERT_EQ(res.terms.size(), 1u);
    EXPECT_FALSE(res.terms[0].is_quadratic);
    EXPECT_NEAR(res.terms[0].r, 1.0, 1e-6);
    EXPECT_NEAR(res.terms[0].A, 1.0, 1e-6);
    EXPECT_EQ(res.terms[0].k, 1);
}

TEST(PolyPartialFractions, ImproperQuadraticHasQuotient) {
    // x^2 / (x - 1) = x + 1 + 1/(x - 1)
    const std::vector<double> num{0.0, 0.0, 1.0};
    const std::vector<double> den{-1.0, 1.0};
    const auto res = poly_partial_fractions(num, den);
    ASSERT_FALSE(res.quotient.empty());
    expect_poly_near(res.quotient, {1.0, 1.0}, 1e-6);
    ASSERT_EQ(res.terms.size(), 1u);
    EXPECT_FALSE(res.terms[0].is_quadratic);
    EXPECT_NEAR(res.terms[0].r, 1.0, 1e-6);
    EXPECT_EQ(res.terms[0].k, 1);
}

// ---------------------------------------------------------------------------
// poly_fit
// ---------------------------------------------------------------------------

TEST(PolyFit, RecoversQuadratic) {
    const std::vector<double> xs{0.0, 1.0, 2.0, 3.0};
    std::vector<double> ys;
    for (double x : xs) {
        ys.push_back(1.0 + 2.0 * x + 3.0 * x * x);
    }
    const auto c = poly_fit(xs, ys, 2);
    expect_poly_near(c, {1.0, 2.0, 3.0}, 1e-8);
    for (size_t i = 0; i < xs.size(); ++i) {
        EXPECT_NEAR(eval_at(c, xs[i]), ys[i], 1e-8) << "i=" << i;
    }
}

TEST(PolyFit, DegreeZeroIsMean) {
    const std::vector<double> xs{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> ys{2.0, 4.0, 6.0, 8.0};
    const auto c = poly_fit(xs, ys, 0);
    ASSERT_FALSE(c.empty());
    EXPECT_NEAR(c[0], 5.0, 1e-9);
}

TEST(PolyFit, EmptyXsDegreeZero) {
    const std::vector<double> xs;
    const std::vector<double> ys;
    const auto c = poly_fit(xs, ys, 0);
    ASSERT_EQ(c.size(), 1u);
    EXPECT_NEAR(c[0], 0.0, 1e-15);
}

TEST(PolyExtTest, empty_pow_monic_reverse_shift_scale) {
    const auto pz = poly_pow({}, 3);
    ASSERT_TRUE(pz.has_value());
    expect_poly_near(*pz, {0.0});

    expect_poly_near(poly_monic({}), {0.0});
    expect_poly_near(poly_reverse({}), {0.0});
    expect_poly_near(poly_shift({}, 2.0), {0.0});
    expect_poly_near(poly_scale({}, 3.0), {0.0});
}

TEST(PolyExtTest, empty_lcm_sylvester_discriminant_squarefree) {
    expect_poly_near(poly_lcm({}, {1.0, 1.0}), {0.0});
    expect_poly_near(poly_lcm({1.0, 1.0}, {}), {0.0});
    expect_poly_near(poly_lcm({0.0}, {1.0, 2.0}), {0.0});

    const auto S = poly_sylvester({}, {1.0});
    EXPECT_EQ(S.rows(), 1u);
    EXPECT_EQ(S.cols(), 1u);

    EXPECT_NEAR(poly_discriminant({5.0}), 0.0, 1e-15);
    EXPECT_NEAR(poly_discriminant({}), 0.0, 1e-15);

    expect_poly_near(poly_squarefree({}), {0.0});
    expect_poly_near(poly_squarefree({4.0}), {4.0});
}

TEST(PolyExtTest, gcd_two_constants_is_one) {
    const auto g = poly_gcd({6.0}, {15.0});
    ASSERT_FALSE(g.empty());
    EXPECT_NEAR(g[0], 1.0, 1e-12);
}

TEST(PolyExtTest, bernstein_invalid_indices) {
    EXPECT_NEAR(bernstein(-1, 0, 0.5), 0.0, 1e-15);
    EXPECT_NEAR(bernstein(3, -1, 0.5), 0.0, 1e-15);
    EXPECT_NEAR(bernstein(2, 3, 0.4), 0.0, 1e-15);
}

TEST(PolyExtTest, roots_empty_constant_and_linear) {
    EXPECT_TRUE(poly_roots({}).empty());
    EXPECT_TRUE(poly_roots({7.0}).empty());
    const auto r = poly_roots({-6.0, 2.0});
    ASSERT_EQ(r.size(), 1u);
    EXPECT_NEAR(r[0].real(), 3.0, 1e-12);
    EXPECT_NEAR(r[0].imag(), 0.0, 1e-12);
}

TEST(PolyExtTest, div_quot_and_mod_remainder) {
    // x^2 - 2 = (x - 1)(x + 1) + (-1);  { -2, 0, 1 } / { -1, 1 }
    const std::vector<double> num{-2.0, 0.0, 1.0};
    const std::vector<double> den{-1.0, 1.0};
    const auto q = poly_div_quot(num, den);
    const auto r = poly_mod(num, den);
    ASSERT_FALSE(q.empty());
    ASSERT_FALSE(r.empty());
    const auto recon = poly_add(poly_mul(q, den), r);
    expect_poly_near(recon, num, 1e-10);

    const auto q_low = poly_div_quot({1.0}, {1.0, 1.0});
    expect_poly_near(q_low, {0.0});
    const auto r_low = poly_mod({3.0, 4.0}, {1.0, 2.0, 3.0});
    expect_poly_near(r_low, {3.0, 4.0});
}

TEST(PolyExtTest, integ_empty_and_compose_empty) {
    const auto ip = poly_integ({}, 5.0);
    expect_poly_near(ip, {5.0});
    expect_poly_near(poly_compose({}, {1.0, 1.0}), {0.0});
    const auto pq = poly_compose({1.0, 2.0}, {0.0, 1.0});
    expect_poly_near(pq, {1.0, 2.0});
}

TEST(PolyExtTest, eval_at_named_span_and_empty) {
    const std::vector<double> coeffs{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> xs{0.0, 1.0, -1.0, 2.0};
    const auto vals = poly_eval_at(coeffs, std::span<const double>(xs));
    ASSERT_EQ(vals.size(), 4u);
    EXPECT_NEAR(vals[0], 1.0, 1e-12);
    EXPECT_NEAR(vals[1], 10.0, 1e-12);
    const std::vector<double> none;
    EXPECT_TRUE(poly_eval_at(coeffs, std::span<const double>(none)).empty());
}

TEST(PolyExtTest, cheb_expand_neg_zero_and_root_count) {
    EXPECT_TRUE(poly_cheb_expand([](double) { return 1.0; }, -1).empty());
    const auto c0 = poly_cheb_expand([](double x) { return x + 3.0; }, 0);
    ASSERT_EQ(c0.size(), 1u);
    EXPECT_NEAR(c0[0], 3.0, 1e-9);

    const std::vector<double> p{2.0, -3.0, 1.0}; // (x-1)(x-2)
    EXPECT_EQ(poly_root_count(p, 0.0, 3.0), 2);
    EXPECT_EQ(poly_root_count(p, 3.0, 4.0), 0);
}

TEST(PolyPartialFractions, ConstantDenomIsQuotient) {
    const std::vector<double> num{2.0, 4.0};
    const std::vector<double> den{2.0};
    const auto res = poly_partial_fractions(num, den);
    expect_poly_near(res.quotient, {1.0, 2.0}, 1e-12);
    EXPECT_TRUE(res.terms.empty());
}

TEST(PolyPartialFractions, EmptyNumeratorProper) {
    const auto res = poly_partial_fractions({}, {-1.0, 1.0});
    EXPECT_TRUE(res.quotient.empty());
}

TEST(PolyExtTest, lagrange_two_points_line) {
    const std::vector<double> xs{0.0, 1.0};
    const std::vector<double> ys{3.0, 5.0};
    const auto c = poly_lagrange(xs, ys);
    ASSERT_GE(c.size(), 1u);
    EXPECT_NEAR(eval_at(c, 0.0), 3.0, 1e-8);
    EXPECT_NEAR(eval_at(c, 1.0), 5.0, 1e-8);
}

TEST(PolyExtTest, roots_quadratic_complex_pair) {
    // x^2 + 1 = 0, constant-first coeffs {1, 0, 1} -> roots ±i
    const auto r = poly_roots({1.0, 0.0, 1.0});
    ASSERT_EQ(r.size(), 2u);
    const double im0 = std::abs(r[0].imag());
    const double im1 = std::abs(r[1].imag());
    EXPECT_NEAR(r[0].real(), 0.0, 1e-10);
    EXPECT_NEAR(r[1].real(), 0.0, 1e-10);
    EXPECT_NEAR(im0, 1.0, 1e-10);
    EXPECT_NEAR(im1, 1.0, 1e-10);
    EXPECT_LT(r[0].imag() * r[1].imag(), 0.0);
}

TEST(PolyExtTest, roots_cubic_one_real_and_complex_pair) {
    // (x-1)(x^2+1) = x^3 - x^2 + x - 1
    const auto r = poly_roots({-1.0, 1.0, -1.0, 1.0});
    ASSERT_EQ(r.size(), 3u);
    int n_real = 0;
    int n_complex = 0;
    for (const auto& z : r) {
        if (std::abs(z.imag()) < 1e-8) {
            ++n_real;
            EXPECT_NEAR(z.real(), 1.0, 1e-6);
        } else {
            ++n_complex;
            EXPECT_NEAR(z.real(), 0.0, 1e-6);
            EXPECT_NEAR(std::abs(z.imag()), 1.0, 1e-6);
        }
    }
    EXPECT_EQ(n_real, 1);
    EXPECT_EQ(n_complex, 2);
}

TEST(PolyExtTest, roots_quartic_two_complex_pairs) {
    // (x^2+1)(x^2+4) = x^4 + 5x^2 + 4 — even degree, no real roots, so the
    // companion QR fallback extracts 2x2 blocks with negative discriminant.
    const auto r = poly_roots({4.0, 0.0, 5.0, 0.0, 1.0});
    ASSERT_EQ(r.size(), 4u);
    int n_complex = 0;
    for (const auto& z : r) {
        if (std::abs(z.imag()) > 1e-6) {
            ++n_complex;
            EXPECT_NEAR(z.real(), 0.0, 1e-4);
        }
    }
    EXPECT_GE(n_complex, 2);
}

TEST(PolyExtTest, zero_poly_early_returns) {
    const auto pz = poly_pow({0.0}, 4);
    ASSERT_TRUE(pz.has_value());
    expect_poly_near(*pz, {0.0});

    expect_poly_near(poly_monic({0.0, 0.0}), {0.0});
    expect_poly_near(poly_reverse({0.0, 0.0}), {0.0});
    expect_poly_near(poly_shift({0.0, 0.0}, 3.0), {0.0});
    expect_poly_near(poly_scale({0.0, 0.0}, 2.0), {0.0});
    expect_poly_near(poly_lcm({}, {}), {0.0});

    const auto S = poly_sylvester({}, {});
    EXPECT_EQ(S.rows(), 1u);
    EXPECT_EQ(S.cols(), 1u);

    EXPECT_NEAR(poly_discriminant({0.0, 0.0}), 0.0, 1e-15);
    expect_poly_near(poly_squarefree({0.0, 0.0}), {0.0});
    EXPECT_TRUE(poly_roots({0.0}).empty());
    EXPECT_TRUE(poly_roots({0.0, 0.0, 0.0}).empty());
}

TEST(PolyExtTest, roots_quartic_all_real_companion_2x2) {
    // (x-1)(x-2)(x-3)(x-4) = x^4 - 10x^3 + 35x^2 - 50x + 24
    // Companion QR / Schur fallback 2x2 blocks have nonnegative discriminant.
    const std::vector<double> four_real{24.0, -50.0, 35.0, -10.0, 1.0};
    const auto r = poly_roots(four_real);
    ASSERT_EQ(r.size(), 4u);
    std::vector<double> reals;
    for (const auto& z : r) {
        EXPECT_NEAR(z.imag(), 0.0, 1e-6);
        reals.push_back(z.real());
    }
    std::sort(reals.begin(), reals.end());
    EXPECT_NEAR(reals[0], 1.0, 1e-5);
    EXPECT_NEAR(reals[1], 2.0, 1e-5);
    EXPECT_NEAR(reals[2], 3.0, 1e-5);
    EXPECT_NEAR(reals[3], 4.0, 1e-5);

    const std::vector<double> repeated{1.0, -4.0, 6.0, -4.0, 1.0};
    const auto rr = poly_roots(repeated);
    ASSERT_EQ(rr.size(), 4u);
    for (const auto& z : rr) {
        EXPECT_NEAR(z.real(), 1.0, 5e-3);
        EXPECT_NEAR(z.imag(), 0.0, 5e-3);
    }
}

TEST(PolyExtTest, sylvester_zero_P_Q_and_eval_empty) {
    const std::vector<double> zero_p{0.0};
    const std::vector<double> zero_q{0.0};
    const auto S00 = poly_sylvester(zero_p, zero_q);
    EXPECT_EQ(S00.rows(), 1u);
    EXPECT_EQ(S00.cols(), 1u);

    const std::vector<double> lin{1.0, 2.0};
    const auto Sp0 = poly_sylvester(zero_p, lin);
    EXPECT_GE(Sp0.rows(), 1u);
    const auto S0q = poly_sylvester(lin, zero_q);
    EXPECT_GE(S0q.rows(), 1u);

    const std::vector<double> empty_p;
    EXPECT_NEAR(poly_eval(empty_p, 3.0)[0], 0.0, 1e-15);
    const std::vector<double> xs{1.0, 2.0};
    const auto at = poly_eval_at(empty_p, std::span<const double>(xs));
    ASSERT_EQ(at.size(), 2u);
    EXPECT_NEAR(at[0], 0.0, 1e-15);
    EXPECT_NEAR(at[1], 0.0, 1e-15);
    const std::vector<double> none;
    EXPECT_TRUE(poly_eval_at(empty_p, std::span<const double>(none)).empty());
    EXPECT_NEAR(poly_cheb_eval(empty_p, 0.5), 0.0, 1e-15);
}

TEST(PolyExtTest, add_mul_deriv_gcd_empty_and_resultant) {
    const std::vector<double> empty_p;
    const std::vector<double> lin{1.0, 2.0};
    const auto sum = poly_add(empty_p, lin);
    expect_poly_near(sum, lin);
    EXPECT_TRUE(poly_mul(empty_p, lin).empty());
    expect_poly_near(poly_deriv(empty_p), {0.0});

    const auto g = poly_gcd(empty_p, lin);
    ASSERT_FALSE(g.empty());
    for (double v : g) EXPECT_TRUE(std::isfinite(v));
    EXPECT_NEAR(poly_resultant(empty_p, lin), 0.0, 1e-12);
}

TEST(PolyExtTest, cheb_eval_t0_expand_linear_and_root_count_reversed) {
    const std::vector<double> t0{1.0};
    EXPECT_NEAR(poly_cheb_eval(t0, 0.3), 1.0, 1e-12);

    const auto c1 = poly_cheb_expand([](double x) { return x; }, 1);
    ASSERT_EQ(c1.size(), 2u);
    EXPECT_NEAR(poly_cheb_eval(c1, 0.5), 0.5, 1e-9);
    EXPECT_NEAR(poly_cheb_eval(c1, -0.25), -0.25, 1e-9);

    const std::vector<double> p{2.0, -3.0, 1.0};
    EXPECT_EQ(poly_root_count(p, 3.0, 0.0), -2);
}

TEST(PolyExtTest, roots_degree5_companion_complex_pairs) {
    // (x-1)(x^2+1)(x^2+4) = x^5 - x^4 + 5x^3 - 5x^2 + 4x - 4
    // Degree > 2 forces companion / Schur 2x2 blocks (disc<0 and a real 1x1).
    const std::vector<double> coeffs{-4.0, 4.0, -5.0, 5.0, -1.0, 1.0};
    const auto r = poly_roots(coeffs);
    ASSERT_EQ(r.size(), 5u);
    int n_real = 0;
    int n_complex = 0;
    for (const auto& z : r) {
        if (std::abs(z.imag()) < 1e-6) {
            ++n_real;
            EXPECT_NEAR(z.real(), 1.0, 1e-4);
        } else {
            ++n_complex;
            EXPECT_NEAR(z.real(), 0.0, 1e-4);
            const double im = std::abs(z.imag());
            EXPECT_TRUE(std::abs(im - 1.0) < 1e-3 || std::abs(im - 2.0) < 1e-3);
        }
    }
    EXPECT_EQ(n_real, 1);
    EXPECT_EQ(n_complex, 4);
}

TEST(PolyPartialFractions, RepeatedComplexPairClusters) {
    // 1 / (x^2+1)^2 : repeated conjugate pair groups in pf_group_roots.
    const std::vector<double> num{1.0};
    const std::vector<double> den{1.0, 0.0, 2.0, 0.0, 1.0};
    const auto res = poly_partial_fractions(num, den);
    EXPECT_TRUE(res.quotient.empty());
    ASSERT_FALSE(res.terms.empty());
    int quad_terms = 0;
    int max_k = 0;
    for (const auto& term : res.terms) {
        EXPECT_TRUE(std::isfinite(term.A));
        EXPECT_TRUE(std::isfinite(term.B));
        EXPECT_TRUE(std::isfinite(term.C));
        if (term.is_quadratic) {
            ++quad_terms;
            max_k = std::max(max_k, term.k);
            EXPECT_NEAR(term.p, 0.0, 5e-3);
            EXPECT_NEAR(term.q, 1.0, 5e-3);
        }
    }
    EXPECT_GE(quad_terms, 1);
    EXPECT_GE(max_k, 1);
}

TEST(PolyPartialFractions, TripleComplexPairOrUnknownsMismatch) {
    // (x^2+1)^3 = x^6 + 3x^4 + 3x^2 + 1. Nearby ±i roots cluster (1101-1107);
    // if grouping count != 6 the solver returns the quotient only (1196).
    const std::vector<double> num{1.0};
    const std::vector<double> den{1.0, 0.0, 3.0, 0.0, 3.0, 0.0, 1.0};
    const auto res = poly_partial_fractions(num, den);
    for (double c : res.quotient) {
        EXPECT_TRUE(std::isfinite(c));
    }
    for (const auto& term : res.terms) {
        EXPECT_TRUE(std::isfinite(term.A));
        EXPECT_TRUE(std::isfinite(term.B));
        EXPECT_TRUE(std::isfinite(term.C));
    }
    if (!res.terms.empty()) {
        bool any_quad = false;
        for (const auto& term : res.terms) {
            if (term.is_quadratic) {
                any_quad = true;
                EXPECT_NEAR(term.q, 1.0, 5e-3);
            }
        }
        EXPECT_TRUE(any_quad);
    }
}

TEST(PolyFit, EmptyAndCollinearHitsSingularDiag) {
    const std::vector<double> empty_xs;
    const std::vector<double> empty_ys;
    const auto c_empty = poly_fit(empty_xs, empty_ys, 2);
    ASSERT_EQ(c_empty.size(), 3u);
    EXPECT_NEAR(c_empty[0], 0.0, 1e-15);
    EXPECT_NEAR(c_empty[1], 0.0, 1e-15);
    EXPECT_NEAR(c_empty[2], 0.0, 1e-15);

    // Repeated x: Vandermonde is rank-1 so Gaussian skip/zero-diag back-sub.
    const std::vector<double> xs{1.0, 1.0, 1.0};
    const std::vector<double> ys{2.0, 3.0, 4.0};
    const auto c = poly_fit(xs, ys, 2);
    ASSERT_EQ(c.size(), 3u);
    for (double v : c) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(PolyFit, DegreeOneAndUnderdeterminedSingular) {
    const std::vector<double> xs{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> ys{1.0, 3.0, 5.0, 7.0};
    const auto line = poly_fit(xs, ys, 1);
    expect_poly_near(line, {1.0, 2.0}, 1e-9);

    const std::vector<double> two_x{0.0, 1.0};
    const std::vector<double> two_y{1.0, 2.0};
    const auto under = poly_fit(two_x, two_y, 4);
    ASSERT_EQ(under.size(), 5u);
    for (double v : under) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(PolyPartialFractions, TwoRealPolesAndRepeatedReal) {
    // 1 / ((x-1)(x-2)) = 1/(x-1) - 1/(x-2)
    const std::vector<double> num{1.0};
    const std::vector<double> den{2.0, -3.0, 1.0};
    const auto two = poly_partial_fractions(num, den);
    EXPECT_TRUE(two.quotient.empty());
    ASSERT_EQ(two.terms.size(), 2u);
    std::vector<double> residues;
    for (const auto& term : two.terms) {
        EXPECT_FALSE(term.is_quadratic);
        EXPECT_EQ(term.k, 1);
        residues.push_back(term.A);
        EXPECT_TRUE(std::abs(term.r - 1.0) < 1e-4 || std::abs(term.r - 2.0) < 1e-4);
    }
    EXPECT_NEAR(std::abs(residues[0]), 1.0, 1e-4);
    EXPECT_NEAR(std::abs(residues[1]), 1.0, 1e-4);
    EXPECT_LT(residues[0] * residues[1], 0.0);

    // 1 / (x-1)^2
    const std::vector<double> repeated_den{1.0, -2.0, 1.0};
    const auto rep = poly_partial_fractions(num, repeated_den);
    EXPECT_TRUE(rep.quotient.empty());
    ASSERT_GE(rep.terms.size(), 1u);
    int max_k = 0;
    for (const auto& term : rep.terms) {
        EXPECT_FALSE(term.is_quadratic);
        EXPECT_NEAR(term.r, 1.0, 1e-3);
        EXPECT_TRUE(std::isfinite(term.A));
        max_k = std::max(max_k, term.k);
    }
    EXPECT_GE(max_k, 1);
}

TEST(PolyPartialFractions, MixedRealAndQuadratic) {
    // 1 / ((x-1)(x^2+1)) = x^3 - x^2 + x - 1
    const std::vector<double> num{1.0};
    const std::vector<double> den{-1.0, 1.0, -1.0, 1.0};
    const auto res = poly_partial_fractions(num, den);
    EXPECT_TRUE(res.quotient.empty());
    ASSERT_FALSE(res.terms.empty());
    int n_real = 0;
    int n_quad = 0;
    for (const auto& term : res.terms) {
        EXPECT_TRUE(std::isfinite(term.A));
        EXPECT_TRUE(std::isfinite(term.B));
        EXPECT_TRUE(std::isfinite(term.C));
        if (term.is_quadratic) {
            ++n_quad;
            EXPECT_NEAR(term.p, 0.0, 5e-3);
            EXPECT_NEAR(term.q, 1.0, 5e-3);
        } else {
            ++n_real;
            EXPECT_NEAR(term.r, 1.0, 1e-3);
        }
    }
    EXPECT_EQ(n_real, 1);
    EXPECT_EQ(n_quad, 1);
}

TEST(PolyExtTest, roots_trailing_zeros_strip_to_quadratic) {
    const std::vector<double> padded{2.0, -3.0, 1.0, 0.0, 0.0};
    const auto r = poly_roots(padded);
    ASSERT_EQ(r.size(), 2u);
    std::vector<double> reals;
    for (const auto& z : r) {
        EXPECT_NEAR(z.imag(), 0.0, 1e-8);
        reals.push_back(z.real());
    }
    std::sort(reals.begin(), reals.end());
    EXPECT_NEAR(reals[0], 1.0, 1e-8);
    EXPECT_NEAR(reals[1], 2.0, 1e-8);
}

TEST(PolyExtTest, cheb_expand_custom_interval_and_null_fn) {
    const auto c = poly_cheb_expand([](double x) { return 2.0 * x + 1.0; }, 1, 0.0, 2.0);
    ASSERT_EQ(c.size(), 2u);
    const double t = (2.0 * 1.5 - 2.0) / 2.0;
    EXPECT_NEAR(poly_cheb_eval(c, t), 4.0, 1e-8);

    std::function<double(double)> none;
    EXPECT_TRUE(poly_cheb_expand(none, 3).empty());
}

TEST(PolyExtTest, roots_degree6_companion_all_real) {
    // (x-1)...(x-6) forces companion QR / 2x2 deflation on a larger Hessenberg.
    const std::vector<double> six{
        -720.0, 1764.0, -1624.0, 735.0, -175.0, 21.0, -1.0};
    const auto r = poly_roots(six);
    ASSERT_EQ(r.size(), 6u);
    std::vector<double> reals;
    for (const auto& z : r) {
        EXPECT_NEAR(z.imag(), 0.0, 2e-3);
        reals.push_back(z.real());
    }
    std::sort(reals.begin(), reals.end());
    for (int k = 0; k < 6; ++k) {
        EXPECT_NEAR(reals[static_cast<size_t>(k)], static_cast<double>(k + 1), 5e-3);
    }
}
