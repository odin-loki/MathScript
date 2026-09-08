// Regression tests for three audited poly defects. Each of these fails against
// the pre-fix code:
//   * poly_roots returned all-zero roots for the entire family x^n +/- c,
//     because the companion matrix of x^n + c is orthogonal and a single-shift
//     QR step is a fixed point on it; after the iteration cap the unconverged
//     Hessenberg diagonal was returned with no error.
//   * poly_lagrange delegated to a normal-equations least-squares solve and so
//     stopped interpolating its own nodes.
//   * poly_fit read ys out of bounds on a size mismatch.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "ms/poly/poly.hpp"

using namespace ms::poly;

namespace {

// Horner evaluation in the module's ascending-power convention.
std::complex<double> eval_complex(const std::vector<double>& c, std::complex<double> x) {
    std::complex<double> v{0.0, 0.0};
    for (std::size_t i = c.size(); i-- > 0;) {
        v = v * x + c[i];
    }
    return v;
}

// The only property a root finder must satisfy: p vanishes at every root it
// returns. Checking residuals rather than pinned values keeps the test honest
// about ordering, which is not part of the contract.
double worst_residual(const std::vector<double>& c) {
    const auto roots = poly_roots(c);
    double worst = 0.0;
    for (const auto& r : roots) {
        worst = std::max(worst, std::abs(eval_complex(c, r)));
    }
    return worst;
}

} // namespace

TEST(PolyRootsAudit, CubeRootsOfMinusOne) {
    // x^3 + 1: the pre-fix code returned {0, 0, 0}, whose residual is 1.0.
    const std::vector<double> p{1.0, 0.0, 0.0, 1.0};
    const auto roots = poly_roots(p);
    ASSERT_EQ(roots.size(), 3u);
    EXPECT_LT(worst_residual(p), 1e-12);

    // One real root at -1 and a conjugate pair at 1/2 +/- i*sqrt(3)/2.
    int real_negative_one = 0;
    int conjugate_pair = 0;
    for (const auto& r : roots) {
        if (std::abs(r.real() + 1.0) < 1e-9 && std::abs(r.imag()) < 1e-9) {
            ++real_negative_one;
        }
        if (std::abs(r.real() - 0.5) < 1e-9 &&
            std::abs(std::abs(r.imag()) - std::sqrt(3.0) / 2.0) < 1e-9) {
            ++conjugate_pair;
        }
    }
    EXPECT_EQ(real_negative_one, 1);
    EXPECT_EQ(conjugate_pair, 2);
}

TEST(PolyRootsAudit, XToTheNPlusConstantFamily) {
    // The whole family the orthogonal-companion fixed point used to swallow.
    EXPECT_LT(worst_residual({1.0, 0.0, 0.0, 1.0}), 1e-12);           // x^3 + 1
    EXPECT_LT(worst_residual({-1.0, 0.0, 0.0, 1.0}), 1e-12);          // x^3 - 1
    EXPECT_LT(worst_residual({-8.0, 0.0, 0.0, 1.0}), 1e-11);          // x^3 - 8
    EXPECT_LT(worst_residual({5.0, 0.0, 0.0, 1.0}), 1e-11);           // x^3 + 5
    EXPECT_LT(worst_residual({1.0, 0.0, 0.0, 0.0, 1.0}), 1e-12);      // x^4 + 1
    EXPECT_LT(worst_residual({2.0, 0.0, 0.0, 0.0, 1.0}), 1e-11);      // x^4 + 2
    EXPECT_LT(worst_residual({-3.0, 0.0, 0.0, 0.0, 0.0, 1.0}), 1e-11); // x^5 - 3
}

TEST(PolyRootsAudit, RealRootsAreExact) {
    // (x-1)(x-2)(x-3) = -6 + 11x - 6x^2 + x^3
    const std::vector<double> p{-6.0, 11.0, -6.0, 1.0};
    auto roots = poly_roots(p);
    ASSERT_EQ(roots.size(), 3u);
    std::vector<double> re;
    for (const auto& r : roots) {
        EXPECT_NEAR(r.imag(), 0.0, 1e-12);
        re.push_back(r.real());
    }
    std::sort(re.begin(), re.end());
    EXPECT_NEAR(re[0], 1.0, 1e-9);
    EXPECT_NEAR(re[1], 2.0, 1e-9);
    EXPECT_NEAR(re[2], 3.0, 1e-9);
}

TEST(PolyRootsAudit, ZeroRootsArePeeled) {
    // x^3 - x = x(x-1)(x+1): a root at the origin plus +/-1.
    const std::vector<double> p{0.0, -1.0, 0.0, 1.0};
    auto roots = poly_roots(p);
    ASSERT_EQ(roots.size(), 3u);
    EXPECT_LT(worst_residual(p), 1e-12);
    std::vector<double> re;
    for (const auto& r : roots) {
        re.push_back(r.real());
    }
    std::sort(re.begin(), re.end());
    EXPECT_NEAR(re[0], -1.0, 1e-9);
    EXPECT_NEAR(re[1], 0.0, 1e-9);
    EXPECT_NEAR(re[2], 1.0, 1e-9);
}

TEST(PolyRootsAudit, FactorRecoversTheCubic) {
    // poly_factor is documented as building on poly_roots; with the old root
    // finder it turned x^3 + 1 into x^3.
    const auto factors = poly_factor({1.0, 0.0, 0.0, 1.0});
    ASSERT_FALSE(factors.empty());
    // Multiplying the factors back together must reproduce the input.
    std::vector<double> product{1.0};
    for (const auto& f : factors) {
        for (int rep = 0; rep < f.multiplicity; ++rep) {
            product = poly_mul(product, f.coeffs);
        }
    }
    ASSERT_GE(product.size(), 4u);
    EXPECT_NEAR(product[0], 1.0, 1e-9);
    EXPECT_NEAR(product[1], 0.0, 1e-9);
    EXPECT_NEAR(product[2], 0.0, 1e-9);
    EXPECT_NEAR(product[3], 1.0, 1e-9);
}

TEST(PolyLagrangeAudit, InterpolatesItsOwnNodes) {
    // 14 equispaced nodes with alternating 0/1 data: the normal-equations route
    // missed its own nodes by up to 0.56 here.
    std::vector<double> xs;
    std::vector<double> ys;
    for (int i = 1; i <= 14; ++i) {
        xs.push_back(static_cast<double>(i));
        ys.push_back(static_cast<double>(i % 2));
    }
    const auto c = poly_lagrange(xs, ys);
    ASSERT_EQ(c.size(), xs.size());
    double worst = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        worst = std::max(worst, std::abs(poly_eval(c, xs[i])[0] - ys[i]));
    }
    EXPECT_LT(worst, 1e-4);
}

TEST(PolyLagrangeAudit, AgreesWithNewtonAsDocumented) {
    // interp_newton's doc comment asserts the two produce the same polynomial.
    const std::vector<double> xs{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> ys{1.0, 3.0, 2.0, 5.0, 4.0};
    const auto a = poly_lagrange(xs, ys);
    const auto b = interp_newton(xs, ys);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], b[i], 1e-9) << "coefficient " << i;
    }
}

TEST(PolyLagrangeAudit, RejectsDegenerateInput) {
    EXPECT_TRUE(poly_lagrange({1.0, 2.0, 2.0}, {1.0, 2.0, 3.0}).empty());  // duplicate node
    EXPECT_TRUE(poly_lagrange({1.0, 2.0, 3.0}, {1.0, 2.0}).empty());       // size mismatch
    EXPECT_TRUE(poly_lagrange({}, {}).empty());
}

TEST(PolyFitAudit, RejectsMismatchedInput) {
    // Previously indexed ys[i] for every i < xs.size().
    EXPECT_TRUE(poly_fit({1.0, 2.0, 3.0}, {1.0, 2.0}, 1).empty());
    EXPECT_TRUE(poly_fit({1.0, 2.0}, {1.0, 2.0}, -1).empty());
    // Empty-but-matching input keeps its existing meaning: degree+1 zeros.
    EXPECT_EQ(poly_fit({}, {}, 2).size(), 3u);
}

TEST(PolyFitAudit, StillFitsAStraightLine) {
    // The guard must not disturb the ordinary path: y = 2x + 1.
    const auto c = poly_fit({0.0, 1.0, 2.0, 3.0}, {1.0, 3.0, 5.0, 7.0}, 1);
    ASSERT_EQ(c.size(), 2u);
    EXPECT_NEAR(c[0], 1.0, 1e-9);
    EXPECT_NEAR(c[1], 2.0, 1e-9);
}
