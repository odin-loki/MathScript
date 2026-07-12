// Wave 56: Advanced polynomial tests — roots, gcd, compose, integ, fit, lagrange
#include "ms/poly/poly.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

// Helpers
static double eval_at(const std::vector<double>& c, double x) {
    return ms::poly::poly_eval(c, x)[0];
}

// -----------------------------------------------------------------------
// poly_div_quot / poly_mod
// -----------------------------------------------------------------------
TEST(PolyDivQuot, LinearByLinear) {
    // (x^2 - 1) / (x + 1) = x - 1  (coeffs low-to-high: {-1, 1})
    std::vector<double> a = {-1.0, 0.0, 1.0};  // x^2 - 1
    std::vector<double> b = {1.0, 1.0};          // x + 1
    auto q = ms::poly::poly_div_quot(a, b);
    EXPECT_EQ(q.size(), 2u);
    EXPECT_NEAR(q[0], -1.0, 1e-10);
    EXPECT_NEAR(q[1],  1.0, 1e-10);
}

TEST(PolyMod, ZeroRemainder) {
    std::vector<double> a = {-1.0, 0.0, 1.0};  // x^2 - 1
    std::vector<double> b = {1.0, 1.0};          // x + 1
    auto r = ms::poly::poly_mod(a, b);
    EXPECT_NEAR(std::abs(r.empty() ? 0.0 : r[0]), 0.0, 1e-10);
}

TEST(PolyMod, NonZeroRemainder) {
    // x^2 / (x + 2) => rem = 4 (since x^2 = (x-2)(x+2) + 4)
    std::vector<double> a = {0.0, 0.0, 1.0};  // x^2
    std::vector<double> b = {2.0, 1.0};         // x + 2
    auto r = ms::poly::poly_mod(a, b);
    ASSERT_FALSE(r.empty());
    EXPECT_NEAR(r[0], 4.0, 1e-9);
}

// -----------------------------------------------------------------------
// poly_integ
// -----------------------------------------------------------------------
TEST(PolyInteg, ConstantPoly) {
    // integral of {3} = {0, 3} (i.e. 3x + c)
    auto q = ms::poly::poly_integ({3.0}, 0.0);
    ASSERT_GE(q.size(), 2u);
    EXPECT_NEAR(q[0], 0.0, 1e-14);
    EXPECT_NEAR(q[1], 3.0, 1e-14);
}

TEST(PolyInteg, QuadraticPoly) {
    // integral of x^2 = x^3/3  => coeffs: {c, 0, 0, 1/3}
    auto q = ms::poly::poly_integ({0.0, 0.0, 1.0}, 0.0);
    ASSERT_GE(q.size(), 4u);
    EXPECT_NEAR(q[3], 1.0 / 3.0, 1e-10);
}

TEST(PolyInteg, ConstantOfIntegration) {
    auto q = ms::poly::poly_integ({1.0}, 5.0);
    EXPECT_NEAR(q[0], 5.0, 1e-14);
}

TEST(PolyInteg, DerivOfIntegIsOriginal) {
    std::vector<double> p = {2.0, -3.0, 1.0};
    auto ip = ms::poly::poly_integ(p, 0.0);
    auto d  = ms::poly::poly_deriv(ip);
    ASSERT_EQ(d.size(), p.size());
    for (size_t i = 0; i < p.size(); ++i)
        EXPECT_NEAR(d[i], p[i], 1e-10);
}

// -----------------------------------------------------------------------
// poly_compose
// -----------------------------------------------------------------------
TEST(PolyCompose, LinearLinear) {
    // p(x) = x + 1, q(x) = 2x => p(q(x)) = 2x + 1 = {1, 2}
    std::vector<double> p = {1.0, 1.0};   // 1 + x
    std::vector<double> q = {0.0, 2.0};   // 2x
    auto pq = ms::poly::poly_compose(p, q);
    ASSERT_GE(pq.size(), 2u);
    EXPECT_NEAR(pq[0], 1.0, 1e-10);
    EXPECT_NEAR(pq[1], 2.0, 1e-10);
}

TEST(PolyCompose, SquareOfLinear) {
    // p(x) = x^2 = {0, 0, 1}, q(x) = x + 1 = {1, 1}
    // p(q(x)) = (x+1)^2 = 1 + 2x + x^2 = {1, 2, 1}
    std::vector<double> p = {0.0, 0.0, 1.0};
    std::vector<double> q = {1.0, 1.0};
    auto pq = ms::poly::poly_compose(p, q);
    ASSERT_GE(pq.size(), 3u);
    EXPECT_NEAR(pq[0], 1.0, 1e-10);
    EXPECT_NEAR(pq[1], 2.0, 1e-10);
    EXPECT_NEAR(pq[2], 1.0, 1e-10);
}

TEST(PolyCompose, IsFinite) {
    auto r = ms::poly::poly_compose({1.0, 2.0, 3.0}, {0.5, 1.5});
    for (double v : r) EXPECT_TRUE(std::isfinite(v));
}

// -----------------------------------------------------------------------
// poly_gcd
// -----------------------------------------------------------------------
TEST(PolyGcd, GcdOfSamePoly) {
    std::vector<double> p = {-1.0, 0.0, 1.0};  // x^2 - 1
    auto g = ms::poly::poly_gcd(p, p);
    // GCD of p with itself is p (monic)
    ASSERT_FALSE(g.empty());
    // Should be monic (leading coefficient 1)
    EXPECT_NEAR(g.back(), 1.0, 1e-9);
}

TEST(PolyGcd, GcdOfCoprimes) {
    // gcd(x^2 - 1, x^2 + 1) = 1  (constants)
    std::vector<double> a = {-1.0, 0.0, 1.0};
    std::vector<double> b = {1.0, 0.0, 1.0};
    auto g = ms::poly::poly_gcd(a, b);
    EXPECT_EQ(g.size(), 1u);
    EXPECT_NEAR(g[0], 1.0, 1e-9);
}

TEST(PolyGcd, GcdNontrivial) {
    // gcd(x^2 - 1, x - 1) = x - 1 = {-1, 1}
    std::vector<double> a = {-1.0, 0.0, 1.0};
    std::vector<double> b = {-1.0, 1.0};
    auto g = ms::poly::poly_gcd(a, b);
    ASSERT_GE(g.size(), 2u);
    // Evaluate at root of (x-1): g(1) should be ~0
    EXPECT_NEAR(eval_at(g, 1.0), 0.0, 1e-8);
}

// -----------------------------------------------------------------------
// poly_roots
// -----------------------------------------------------------------------
TEST(PolyRoots, LinearRoot) {
    // p(x) = x - 3 = {-3, 1} => root = 3
    auto roots = ms::poly::poly_roots({-3.0, 1.0});
    ASSERT_EQ(roots.size(), 1u);
    EXPECT_NEAR(roots[0].real(), 3.0, 1e-6);
    EXPECT_NEAR(roots[0].imag(), 0.0, 1e-6);
}

TEST(PolyRoots, QuadraticRealRoots) {
    // p(x) = x^2 - 5x + 6 = (x-2)(x-3) => roots 2, 3
    auto roots = ms::poly::poly_roots({6.0, -5.0, 1.0});
    ASSERT_EQ(roots.size(), 2u);
    // Both roots should be real
    for (auto& r : roots) EXPECT_NEAR(r.imag(), 0.0, 1e-6);
    // Sort and check
    std::vector<double> real_roots = {roots[0].real(), roots[1].real()};
    std::sort(real_roots.begin(), real_roots.end());
    EXPECT_NEAR(real_roots[0], 2.0, 1e-6);
    EXPECT_NEAR(real_roots[1], 3.0, 1e-6);
}

TEST(PolyRoots, QuadraticComplexRoots) {
    // p(x) = x^2 + 1 => roots ±i
    auto roots = ms::poly::poly_roots({1.0, 0.0, 1.0});
    ASSERT_EQ(roots.size(), 2u);
    for (auto& r : roots) {
        EXPECT_NEAR(r.real(), 0.0, 1e-5);
        EXPECT_NEAR(std::abs(r.imag()), 1.0, 1e-5);
    }
}

TEST(PolyRoots, RootsCountMatchesDegree) {
    auto roots = ms::poly::poly_roots({1.0, -2.0, 1.0, -1.0, 0.5});
    EXPECT_EQ(roots.size(), 4u);
}

TEST(PolyRoots, RootsAreRoots) {
    // p(x) = x^2 - 5x + 6
    std::vector<double> p = {6.0, -5.0, 1.0};
    auto roots = ms::poly::poly_roots(p);
    for (auto& r : roots) {
        // |p(r)| should be small
        auto eval = std::complex<double>(p[2]) * r * r +
                    std::complex<double>(p[1]) * r +
                    std::complex<double>(p[0]);
        EXPECT_NEAR(std::abs(eval), 0.0, 1e-5);
    }
}

// -----------------------------------------------------------------------
// poly_fit
// -----------------------------------------------------------------------
TEST(PolyFit, LinearFit) {
    std::vector<double> xs = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> ys = {1.0, 3.0, 5.0, 7.0};  // y = 2x + 1
    auto c = ms::poly::poly_fit(xs, ys, 1);
    ASSERT_GE(c.size(), 2u);
    EXPECT_NEAR(c[0], 1.0, 1e-8);  // intercept
    EXPECT_NEAR(c[1], 2.0, 1e-8);  // slope
}

TEST(PolyFit, QuadraticFit) {
    std::vector<double> xs = {-1.0, 0.0, 1.0, 2.0, 3.0};
    std::vector<double> ys;
    for (double x : xs) ys.push_back(x * x - x + 2.0);  // y = x^2 - x + 2
    auto c = ms::poly::poly_fit(xs, ys, 2);
    ASSERT_GE(c.size(), 3u);
    EXPECT_NEAR(c[0], 2.0, 1e-7);
    EXPECT_NEAR(c[1], -1.0, 1e-7);
    EXPECT_NEAR(c[2], 1.0, 1e-7);
}

TEST(PolyFit, FitIsFinite) {
    std::vector<double> xs = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> ys = {2.0, 4.0, 6.0, 8.0};
    auto c = ms::poly::poly_fit(xs, ys, 2);
    for (double v : c) EXPECT_TRUE(std::isfinite(v));
}

// -----------------------------------------------------------------------
// poly_cheb_eval
// -----------------------------------------------------------------------
TEST(PolyChebEval, T0IsOne) {
    double v = ms::poly::poly_cheb_eval({1.0}, 0.5);
    EXPECT_NEAR(v, 1.0, 1e-12);
}

TEST(PolyChebEval, T1IsX) {
    // coeffs = {0, 1} => T_0 * 0 + T_1 * 1 = x
    double x = 0.7;
    double v = ms::poly::poly_cheb_eval({0.0, 1.0}, x);
    EXPECT_NEAR(v, x, 1e-12);
}

// -----------------------------------------------------------------------
// poly_root_count (Sturm's theorem)
// -----------------------------------------------------------------------
TEST(PolyRootCount, QuadraticTwoRootsInInterval) {
    // x^2 - 5x + 6 has roots 2, 3 — both in [1, 4]
    int n = ms::poly::poly_root_count({6.0, -5.0, 1.0}, 1.0, 4.0);
    EXPECT_EQ(n, 2);
}

TEST(PolyRootCount, NoRootsInInterval) {
    // x^2 + 1 has no real roots
    int n = ms::poly::poly_root_count({1.0, 0.0, 1.0}, -10.0, 10.0);
    EXPECT_EQ(n, 0);
}

TEST(PolyRootCount, OneRootInInterval) {
    // x^2 - 5x + 6, only root 2 is in [1.5, 2.5]
    int n = ms::poly::poly_root_count({6.0, -5.0, 1.0}, 1.5, 2.5);
    EXPECT_EQ(n, 1);
}

// -----------------------------------------------------------------------
// poly_cheb_expand
// -----------------------------------------------------------------------

// Maps an evaluation point x in [a, b] back to the canonical [-1, 1]
// argument expected by poly_cheb_eval.
static double to_canonical(double x, double a, double b) {
    return (2.0 * x - (a + b)) / (b - a);
}

TEST(PolyChebExpand, RoundTripExpOnDefaultInterval) {
    auto f = [](double x) { return std::exp(x); };
    int n = 16;
    auto coeffs = ms::poly::poly_cheb_expand(f, n);
    ASSERT_EQ(coeffs.size(), static_cast<size_t>(n + 1));

    for (double x : {-0.9, -0.5, -0.1, 0.0, 0.3, 0.6, 0.95}) {
        double approx = ms::poly::poly_cheb_eval(coeffs, x);
        EXPECT_NEAR(approx, f(x), 1e-9)
            << "mismatch at x=" << x;
    }
}

TEST(PolyChebExpand, ExactForLowDegreePolynomial) {
    // f(x) = x^3 - 2x + 1; Chebyshev polys T_0..T_n (n=3) exactly span
    // degree-<=3 polynomials, so the expansion should reproduce f exactly.
    auto f = [](double x) { return x * x * x - 2.0 * x + 1.0; };
    int n = 3;
    auto coeffs = ms::poly::poly_cheb_expand(f, n);
    ASSERT_EQ(coeffs.size(), static_cast<size_t>(n + 1));

    for (double x : {-1.0, -0.7, -0.25, 0.0, 0.4, 0.8, 1.0}) {
        double approx = ms::poly::poly_cheb_eval(coeffs, x);
        EXPECT_NEAR(approx, f(x), 1e-9) << "mismatch at x=" << x;
    }

    // Also check with a higher n: still exact, since higher-order Chebyshev
    // terms simply pick up (near-)zero coefficients.
    auto coeffs2 = ms::poly::poly_cheb_expand(f, 8);
    for (double x : {-0.9, -0.3, 0.2, 0.6, 0.99}) {
        double approx = ms::poly::poly_cheb_eval(coeffs2, x);
        EXPECT_NEAR(approx, f(x), 1e-9) << "mismatch at x=" << x;
    }
}

TEST(PolyChebExpand, RoundTripGeneralInterval) {
    // f(x) = sin(x) on [0, 5] -- interval mapping must be correct for the
    // round trip to hold outside the canonical [-1, 1] range.
    auto f = [](double x) { return std::sin(x); };
    double a = 0.0, b = 5.0;
    int n = 18;
    auto coeffs = ms::poly::poly_cheb_expand(f, n, a, b);
    ASSERT_EQ(coeffs.size(), static_cast<size_t>(n + 1));

    for (double x : {0.1, 1.0, 2.5, 3.7, 4.9}) {
        double t = to_canonical(x, a, b);
        double approx = ms::poly::poly_cheb_eval(coeffs, t);
        EXPECT_NEAR(approx, f(x), 1e-8) << "mismatch at x=" << x;
    }
}

TEST(PolyChebExpand, AccuracyImprovesWithN) {
    auto f = [](double x) { return std::sin(x) + 0.3 * std::exp(0.5 * x); };
    std::vector<double> test_pts = {-0.95, -0.6, -0.2, 0.1, 0.5, 0.8, 0.97};

    auto max_err = [&](int n) {
        auto coeffs = ms::poly::poly_cheb_expand(f, n);
        double worst = 0.0;
        for (double x : test_pts) {
            double approx = ms::poly::poly_cheb_eval(coeffs, x);
            worst = std::max(worst, std::abs(approx - f(x)));
        }
        return worst;
    };

    double err_low = max_err(3);
    double err_high = max_err(14);
    EXPECT_LT(err_high, err_low);
    // The high-degree expansion of a smooth function should be very
    // accurate in absolute terms, not merely "better than the low one".
    EXPECT_LT(err_high, 1e-8);
}

TEST(PolyChebExpand, DegreeZeroConstantApproximation) {
    auto f = [](double x) { return x * x; };  // f(0.5*(a+b)) at midpoint of [-1,1] is 0
    auto coeffs = ms::poly::poly_cheb_expand(f, 0);
    ASSERT_EQ(coeffs.size(), 1u);
    // n=0 has exactly one Chebyshev node, x_0 = cos(pi/2) = 0, mapped to the
    // midpoint of [-1,1] (which is 0), so the single coefficient equals
    // f(0), and poly_cheb_eval should reproduce that constant everywhere.
    EXPECT_NEAR(coeffs[0], f(0.0), 1e-12);
    EXPECT_NEAR(ms::poly::poly_cheb_eval(coeffs, 0.3), f(0.0), 1e-12);
    EXPECT_NEAR(ms::poly::poly_cheb_eval(coeffs, -0.7), f(0.0), 1e-12);
}

TEST(PolyChebExpand, SmallNHandledGracefully) {
    auto f = [](double x) { return 1.0 + x; };
    for (int n = 0; n <= 2; ++n) {
        auto coeffs = ms::poly::poly_cheb_expand(f, n);
        ASSERT_EQ(coeffs.size(), static_cast<size_t>(n + 1));
        for (double c : coeffs) EXPECT_TRUE(std::isfinite(c));
    }
}

TEST(PolyChebExpand, NegativeDegreeReturnsEmpty) {
    auto f = [](double x) { return x; };
    auto coeffs = ms::poly::poly_cheb_expand(f, -1);
    EXPECT_TRUE(coeffs.empty());
}

// -----------------------------------------------------------------------
// poly_partial_fractions
// -----------------------------------------------------------------------
namespace {

using ms::poly::PartialFractionResult;
using ms::poly::PartialFractionTerm;

double eval_pf_term(const PartialFractionTerm& t, double x) {
    if (t.is_quadratic) {
        double denom = std::pow(x * x + t.p * x + t.q, t.k);
        return (t.B * x + t.C) / denom;
    }
    double denom = std::pow(x - t.r, t.k);
    return t.A / denom;
}

// Reconstructs quotient(x) + sum of all terms at x, for comparison against
// numerator(x)/denominator(x) evaluated directly.
double eval_pf_result(const PartialFractionResult& res, double x) {
    double v = res.quotient.empty() ? 0.0 : ms::poly::poly_eval(res.quotient, x)[0];
    for (const auto& t : res.terms) v += eval_pf_term(t, x);
    return v;
}

// Checks that poly_partial_fractions's reconstruction matches
// numerator(x)/denominator(x) directly at every sample point (avoiding
// exact pole locations), which is the strongest correctness signal since it
// doesn't depend on any particular grouping/ordering of terms.
void expect_reconstructs(const std::vector<double>& numerator,
                          const std::vector<double>& denominator,
                          const PartialFractionResult& res,
                          const std::vector<double>& sample_xs,
                          double tol = 1e-6) {
    for (double x : sample_xs) {
        double expected = eval_at(numerator, x) / eval_at(denominator, x);
        double actual = eval_pf_result(res, x);
        EXPECT_NEAR(actual, expected, tol) << "mismatch at x=" << x;
    }
}

} // namespace

TEST(PolyPartialFractions, TextbookDistinctRealPoles) {
    // 1 / ((x-1)(x-2)) = -1/(x-1) + 1/(x-2)
    // (x-1)(x-2) = x^2 - 3x + 2 => denominator {2, -3, 1}
    std::vector<double> num = {1.0};
    std::vector<double> den = {2.0, -3.0, 1.0};
    auto res = ms::poly::poly_partial_fractions(num, den);
    EXPECT_TRUE(res.quotient.empty());
    ASSERT_EQ(res.terms.size(), 2u);
    for (const auto& t : res.terms) {
        EXPECT_FALSE(t.is_quadratic);
        EXPECT_EQ(t.k, 1);
        if (std::abs(t.r - 1.0) < 1e-4) {
            EXPECT_NEAR(t.A, -1.0, 1e-6);
        } else {
            ASSERT_NEAR(t.r, 2.0, 1e-4);
            EXPECT_NEAR(t.A, 1.0, 1e-6);
        }
    }
    expect_reconstructs(num, den, res, {0.0, 0.5, 1.5, 3.0, 5.0, -2.0});
}

TEST(PolyPartialFractions, AnotherDistinctRealPolesPair) {
    // 1 / ((x-3)(x+4)) => denominator (x-3)(x+4) = x^2 + x - 12
    std::vector<double> num = {1.0};
    std::vector<double> den = {-12.0, 1.0, 1.0};
    auto res = ms::poly::poly_partial_fractions(num, den);
    ASSERT_EQ(res.terms.size(), 2u);
    // 1/((x-3)(x+4)) = (1/7)/(x-3) - (1/7)/(x+4)
    for (const auto& t : res.terms) {
        EXPECT_FALSE(t.is_quadratic);
        if (std::abs(t.r - 3.0) < 1e-4) {
            EXPECT_NEAR(t.A, 1.0 / 7.0, 1e-6);
        } else {
            ASSERT_NEAR(t.r, -4.0, 1e-4);
            EXPECT_NEAR(t.A, -1.0 / 7.0, 1e-6);
        }
    }
    expect_reconstructs(num, den, res, {0.0, 1.0, -1.0, 5.0, -10.0, 10.0});
}

TEST(PolyPartialFractions, LinearNumeratorOverDistinctPoles) {
    // (2x - 1) / ((x-1)(x-5)); denominator x^2 - 6x + 5
    std::vector<double> num = {-1.0, 2.0};
    std::vector<double> den = {5.0, -6.0, 1.0};
    auto res = ms::poly::poly_partial_fractions(num, den);
    ASSERT_EQ(res.terms.size(), 2u);
    expect_reconstructs(num, den, res, {0.0, 2.0, 3.0, 6.0, -2.0, 10.0});
}

TEST(PolyPartialFractions, RepeatedRealPoleConstructedExactly) {
    // D(x) = (x-1)^2 * (x-2), constructed by explicit polynomial multiplication.
    std::vector<double> factor1 = {-1.0, 1.0}; // x - 1
    std::vector<double> factor2 = {-2.0, 1.0}; // x - 2
    auto den = ms::poly::poly_mul(ms::poly::poly_mul(factor1, factor1), factor2);
    std::vector<double> num = {1.0};
    auto res = ms::poly::poly_partial_fractions(num, den);
    EXPECT_TRUE(res.quotient.empty());
    ASSERT_EQ(res.terms.size(), 3u);

    bool found_k1 = false, found_k2 = false, found_pole2 = false;
    for (const auto& t : res.terms) {
        EXPECT_FALSE(t.is_quadratic);
        if (std::abs(t.r - 1.0) < 1e-3 && t.k == 1) {
            found_k1 = true;
            EXPECT_NEAR(t.A, -1.0, 1e-4);
        } else if (std::abs(t.r - 1.0) < 1e-3 && t.k == 2) {
            found_k2 = true;
            EXPECT_NEAR(t.A, -1.0, 1e-4);
        } else if (std::abs(t.r - 2.0) < 1e-3) {
            found_pole2 = true;
            EXPECT_NEAR(t.A, 1.0, 1e-4);
        }
    }
    EXPECT_TRUE(found_k1);
    EXPECT_TRUE(found_k2);
    EXPECT_TRUE(found_pole2);
    expect_reconstructs(num, den, res, {0.0, 0.5, 1.5, 3.0, -1.0, 5.0}, 1e-4);
}

TEST(PolyPartialFractions, RepeatedRealPoleCubed) {
    // D(x) = (x-2)^3, single pole of multiplicity 3.
    std::vector<double> factor = {-2.0, 1.0}; // x - 2
    auto den = ms::poly::poly_mul(ms::poly::poly_mul(factor, factor), factor);
    std::vector<double> num = {1.0, 1.0}; // x + 1
    auto res = ms::poly::poly_partial_fractions(num, den);
    EXPECT_TRUE(res.quotient.empty());
    ASSERT_EQ(res.terms.size(), 3u);
    for (const auto& t : res.terms) {
        EXPECT_FALSE(t.is_quadratic);
        EXPECT_NEAR(t.r, 2.0, 1e-3);
    }
    expect_reconstructs(num, den, res, {0.0, 1.0, 3.0, 4.0, -1.0, 10.0}, 1e-4);
}

TEST(PolyPartialFractions, TwoDistinctRepeatedRealPoles) {
    // D(x) = (x-1)^2 * (x+3)^2
    std::vector<double> f1 = {-1.0, 1.0}; // x - 1
    std::vector<double> f2 = {3.0, 1.0};  // x + 3
    auto den = ms::poly::poly_mul(ms::poly::poly_mul(f1, f1), ms::poly::poly_mul(f2, f2));
    std::vector<double> num = {2.0, -1.0, 1.0}; // x^2 - x + 2
    auto res = ms::poly::poly_partial_fractions(num, den);
    EXPECT_TRUE(res.quotient.empty());
    ASSERT_EQ(res.terms.size(), 4u);
    expect_reconstructs(num, den, res, {0.0, 2.0, 5.0, -5.0, -0.5, 10.0}, 1e-4);
}

TEST(PolyPartialFractions, ImproperFractionSplitsQuotientAndRemainder) {
    // (x^3 + 1) / ((x-1)(x-2)) : deg(N)=3 >= deg(D)=2, so a polynomial
    // quotient part is expected in addition to the proper-fraction terms.
    std::vector<double> num = {1.0, 0.0, 0.0, 1.0}; // x^3 + 1
    std::vector<double> den = {2.0, -3.0, 1.0};      // (x-1)(x-2)
    auto res = ms::poly::poly_partial_fractions(num, den);
    EXPECT_FALSE(res.quotient.empty());
    ASSERT_EQ(res.terms.size(), 2u);

    // Verify quotient/remainder split directly via poly_div_quot/poly_mod.
    auto expected_quot = ms::poly::poly_div_quot(num, den);
    auto expected_rem = ms::poly::poly_mod(num, den);
    ASSERT_EQ(res.quotient.size(), expected_quot.size());
    for (size_t i = 0; i < res.quotient.size(); ++i) {
        EXPECT_NEAR(res.quotient[i], expected_quot[i], 1e-8);
    }
    // The remainder N_r(x)/D(x) reconstructed from just the terms should
    // match expected_rem(x)/den(x).
    for (double x : {0.0, 3.0, -1.0, 5.0}) {
        double term_sum = 0.0;
        for (const auto& t : res.terms) term_sum += eval_pf_term(t, x);
        double expected = eval_at(expected_rem, x) / eval_at(den, x);
        EXPECT_NEAR(term_sum, expected, 1e-6) << "mismatch at x=" << x;
    }

    expect_reconstructs(num, den, res, {0.0, 3.0, -1.0, 5.0, 10.0});
}

TEST(PolyPartialFractions, ImproperFractionEqualDegree) {
    // (3x^2 + 1) / (x^2 - 1): deg(N) == deg(D), quotient should be a nonzero constant.
    std::vector<double> num = {1.0, 0.0, 3.0};
    std::vector<double> den = {-1.0, 0.0, 1.0}; // (x-1)(x+1)
    auto res = ms::poly::poly_partial_fractions(num, den);
    ASSERT_FALSE(res.quotient.empty());
    EXPECT_NEAR(res.quotient[0], 3.0, 1e-6); // 3x^2+1 = 3*(x^2-1) + 4
    ASSERT_EQ(res.terms.size(), 2u);
    expect_reconstructs(num, den, res, {0.0, 2.0, -2.0, 5.0, -5.0});
}

TEST(PolyPartialFractions, ComplexConjugatePolePair) {
    // 1 / (x^2 + 1) => single quadratic term with p=0, q=1, B=0, C=1.
    std::vector<double> num = {1.0};
    std::vector<double> den = {1.0, 0.0, 1.0};
    auto res = ms::poly::poly_partial_fractions(num, den);
    EXPECT_TRUE(res.quotient.empty());
    ASSERT_EQ(res.terms.size(), 1u);
    const auto& t = res.terms[0];
    EXPECT_TRUE(t.is_quadratic);
    EXPECT_EQ(t.k, 1);
    EXPECT_NEAR(t.p, 0.0, 1e-6);
    EXPECT_NEAR(t.q, 1.0, 1e-6);
    EXPECT_NEAR(t.B, 0.0, 1e-6);
    EXPECT_NEAR(t.C, 1.0, 1e-6);
    expect_reconstructs(num, den, res, {0.0, 1.0, -1.0, 2.5, -3.0});
}

TEST(PolyPartialFractions, ComplexPolePairWithLinearNumerator) {
    // (x + 1) / ((x-1)(x^2+4)); denominator (x-1)(x^2+4) = x^3 - x^2 + 4x - 4
    std::vector<double> lin = {-1.0, 1.0};      // x - 1
    std::vector<double> quad = {4.0, 0.0, 1.0}; // x^2 + 4
    auto den = ms::poly::poly_mul(lin, quad);
    std::vector<double> num = {1.0, 1.0}; // x + 1
    auto res = ms::poly::poly_partial_fractions(num, den);
    EXPECT_TRUE(res.quotient.empty());
    ASSERT_EQ(res.terms.size(), 2u);
    bool found_real = false, found_quad = false;
    for (const auto& t : res.terms) {
        if (!t.is_quadratic) {
            found_real = true;
            EXPECT_NEAR(t.r, 1.0, 1e-4);
        } else {
            found_quad = true;
            EXPECT_NEAR(t.p, 0.0, 1e-4);
            EXPECT_NEAR(t.q, 4.0, 1e-4);
        }
    }
    EXPECT_TRUE(found_real);
    EXPECT_TRUE(found_quad);
    expect_reconstructs(num, den, res, {0.0, 2.0, -2.0, 5.0, -5.0});
}

TEST(PolyPartialFractions, DegenerateDegreeZeroDenominator) {
    // N(x) / c is just N(x)/c, a polynomial quotient with no terms.
    std::vector<double> num = {2.0, 4.0, 6.0}; // 2 + 4x + 6x^2
    std::vector<double> den = {2.0};
    auto res = ms::poly::poly_partial_fractions(num, den);
    EXPECT_TRUE(res.terms.empty());
    ASSERT_GE(res.quotient.size(), 3u);
    EXPECT_NEAR(res.quotient[0], 1.0, 1e-10);
    EXPECT_NEAR(res.quotient[1], 2.0, 1e-10);
    EXPECT_NEAR(res.quotient[2], 3.0, 1e-10);
    expect_reconstructs(num, den, res, {0.0, 1.0, -1.0, 5.0});
}

TEST(PolyPartialFractions, DegenerateDegreeOneDenominator) {
    // 5 / (x - 3): a single simple pole.
    std::vector<double> num = {5.0};
    std::vector<double> den = {-3.0, 1.0};
    auto res = ms::poly::poly_partial_fractions(num, den);
    EXPECT_TRUE(res.quotient.empty());
    ASSERT_EQ(res.terms.size(), 1u);
    EXPECT_FALSE(res.terms[0].is_quadratic);
    EXPECT_NEAR(res.terms[0].r, 3.0, 1e-8);
    EXPECT_NEAR(res.terms[0].A, 5.0, 1e-8);
    expect_reconstructs(num, den, res, {0.0, 1.0, 10.0, -10.0});
}

TEST(PolyPartialFractions, DegenerateDegreeOneWithNegativeSlope) {
    // 3 / (2 - x) = 3 / (-(x - 2)) => denominator {2, -1}
    std::vector<double> num = {3.0};
    std::vector<double> den = {2.0, -1.0};
    auto res = ms::poly::poly_partial_fractions(num, den);
    ASSERT_EQ(res.terms.size(), 1u);
    EXPECT_NEAR(res.terms[0].r, 2.0, 1e-8);
    EXPECT_NEAR(res.terms[0].A, -3.0, 1e-8);
    expect_reconstructs(num, den, res, {0.0, 5.0, -5.0});
}

TEST(PolyPartialFractions, ThreeDistinctRealPolesReconstructs) {
    // D(x) = (x+1)(x-2)(x+5), numerator a generic quadratic.
    std::vector<double> f1 = {1.0, 1.0};  // x + 1
    std::vector<double> f2 = {-2.0, 1.0}; // x - 2
    std::vector<double> f3 = {5.0, 1.0};  // x + 5
    auto den = ms::poly::poly_mul(ms::poly::poly_mul(f1, f2), f3);
    std::vector<double> num = {1.0, 2.0, 3.0}; // 3x^2 + 2x + 1
    auto res = ms::poly::poly_partial_fractions(num, den);
    EXPECT_TRUE(res.quotient.empty());
    ASSERT_EQ(res.terms.size(), 3u);
    for (const auto& t : res.terms) EXPECT_FALSE(t.is_quadratic);
    expect_reconstructs(num, den, res, {0.0, 1.0, -3.0, 3.0, -6.0, 10.0});
}

TEST(PolyPartialFractions, MixedRepeatedAndSimplePole) {
    // D(x) = (x-1)(x-2)^2, mixing a simple pole with a repeated one.
    std::vector<double> f1 = {-1.0, 1.0}; // x - 1
    std::vector<double> f2 = {-2.0, 1.0}; // x - 2
    auto den = ms::poly::poly_mul(f1, ms::poly::poly_mul(f2, f2));
    std::vector<double> num = {1.0, -2.0, 1.0}; // x^2 - 2x + 1
    auto res = ms::poly::poly_partial_fractions(num, den);
    EXPECT_TRUE(res.quotient.empty());
    ASSERT_EQ(res.terms.size(), 3u);
    int max_k_at_2 = 0;
    for (const auto& t : res.terms) {
        EXPECT_FALSE(t.is_quadratic);
        if (std::abs(t.r - 2.0) < 1e-3) max_k_at_2 = std::max(max_k_at_2, t.k);
    }
    EXPECT_EQ(max_k_at_2, 2);
    expect_reconstructs(num, den, res, {0.0, 3.0, -3.0, 5.0, 1.5, 10.0}, 1e-4);
}

TEST(PolyPartialFractions, AllCoefficientsFinite) {
    std::vector<double> f1 = {-1.0, 1.0};
    std::vector<double> f2 = {-4.0, 1.0};
    auto den = ms::poly::poly_mul(f1, f2);
    std::vector<double> num = {1.0, 1.0};
    auto res = ms::poly::poly_partial_fractions(num, den);
    for (const auto& t : res.terms) {
        EXPECT_TRUE(std::isfinite(t.A));
        EXPECT_TRUE(std::isfinite(t.B));
        EXPECT_TRUE(std::isfinite(t.C));
        EXPECT_TRUE(std::isfinite(t.r));
        EXPECT_TRUE(std::isfinite(t.p));
        EXPECT_TRUE(std::isfinite(t.q));
    }
    for (double c : res.quotient) EXPECT_TRUE(std::isfinite(c));
}

TEST(PolyPartialFractions, ZeroDenominatorReturnsEmpty) {
    std::vector<double> num = {1.0};
    std::vector<double> den = {0.0};
    auto res = ms::poly::poly_partial_fractions(num, den);
    EXPECT_TRUE(res.quotient.empty());
    EXPECT_TRUE(res.terms.empty());
}
