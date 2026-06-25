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
