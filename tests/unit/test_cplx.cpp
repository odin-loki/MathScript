#define _USE_MATH_DEFINES
#include "ms/cplx/cplx.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <utility>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms::cplx;

// ---- Residue ----
TEST(CplxResidue, SimplePole) {
    // f(z) = 1/(z-1): residue at z=1 is 1
    auto f = [](C z) { return C(1.0) / (z - C(1.0)); };
    C res = residue(f, C(1.0), 1e-5);
    EXPECT_NEAR(res.real(), 1.0, 1e-3);
    EXPECT_NEAR(res.imag(), 0.0, 1e-3);
}

TEST(CplxResidue, OrderTwo) {
    // f(z) = 1/(z-0)^2 has residue 0 at z=0 (it's a pole of order 2, residue=0)
    // For 1/(z*(z-1)): residue at z=0 is -1
    auto f = [](C z) { return C(1.0) / (z * (z - C(1.0))); };
    C res = residue(f, C(0.0), 1e-5);
    EXPECT_NEAR(res.real(), -1.0, 1e-2);
}

// ---- Winding number ----
TEST(CplxWinding, Circle) {
    // Unit circle winds once around origin
    int N = 100;
    std::vector<C> gamma;
    for (int i = 0; i <= N; ++i) {
        double theta = 2.0 * M_PI * i / N;
        gamma.push_back(C(std::cos(theta), std::sin(theta)));
    }
    EXPECT_EQ(winding_number(gamma, C(0.0)), 1);
    EXPECT_EQ(winding_number(gamma, C(2.0, 0.0)), 0);
}

// ---- Contour integral ----
TEST(CplxContour, Integral) {
    // ∮ dz = 0 for any closed path
    int N = 100;
    std::vector<C> path;
    for (int i = 0; i <= N; ++i) {
        double t = 2.0 * M_PI * i / N;
        path.push_back(C(std::cos(t), std::sin(t)));
    }
    auto f = [](C z) { return z * z; };  // analytic: integral = 0
    C val = contour_integral(f, path, 20);
    EXPECT_NEAR(std::abs(val), 0.0, 0.1);
}

// ---- Möbius ----
TEST(CplxMobius, Composition) {
    Mobius m1(C(1.0), C(0.0), C(0.0), C(1.0));  // identity
    Mobius m2(C(2.0), C(1.0), C(0.0), C(1.0));  // z → 2z+1
    auto comp = m1.compose(m2);
    // Identity composed with m2 should be m2
    EXPECT_NEAR(comp(C(1.0)).real(), m2(C(1.0)).real(), 1e-10);
}

TEST(CplxMobius, Inverse) {
    Mobius m(C(2.0), C(1.0), C(1.0), C(1.0));  // (2z+1)/(z+1)
    auto inv = m.inverse();
    C z(0.5, 0.3);
    C w = m(z);
    C z2 = inv(w);
    EXPECT_NEAR(z2.real(), z.real(), 1e-10);
    EXPECT_NEAR(z2.imag(), z.imag(), 1e-10);
}

// ---- Joukowski ----
TEST(CplxJoukowski, Transform) {
    // z + 1/z at z=i gives i - i = 0... wait: c=1, z=2i: 2i + 1/(2i) = 2i - i/2 = 1.5i
    C w = joukowski(C(0.0, 2.0), 1.0);
    EXPECT_NEAR(w.real(), 0.0, 1e-10);
    EXPECT_NEAR(w.imag(), 1.5, 1e-10);
}

// ---- Cross ratio ----
TEST(CplxCrossRatio, FourPoints) {
    // (z1-z3)(z2-z4) / ((z1-z4)(z2-z3))
    C z1(0.0), z2(1.0), z3(2.0), z4(3.0);
    double cr = cross_ratio(z1, z2, z3, z4);
    // (0-2)(1-3) / ((0-3)(1-2)) = (-2)(-2) / ((-3)(-1)) = 4/3
    EXPECT_NEAR(cr, 4.0 / 3.0, 1e-10);
}

// ---- Hyperbolic distance ----
TEST(CplxHyperbolic, Distance) {
    // Distance from 0 to 0 is 0
    EXPECT_NEAR(hyperbolic_distance(C(0.0), C(0.0)), 0.0, 1e-10);
    // Distance is positive
    double d = hyperbolic_distance(C(0.0), C(0.3, 0.0));
    EXPECT_GT(d, 0.0);
}

// ---- Line integral ----
TEST(CplxLineIntegral, Segment) {
    // ∫_0^1 dz = 1
    auto f = [](C z) { (void)z; return C(1.0); };
    C val = line_integral(f, C(0.0), C(1.0), 50);
    EXPECT_NEAR(val.real(), 1.0, 1e-10);
    EXPECT_NEAR(val.imag(), 0.0, 1e-10);
}

// ---- Power series ----
TEST(CplxPowerSeries, Eval) {
    // exp(z) ≈ 1 + z + z^2/2 + z^3/6 + z^4/24 + z^5/120 + z^6/720 + z^7/5040
    // 8 terms for better accuracy at z=0.5
    std::vector<C> coeffs = {1.0, 1.0, 0.5, 1.0/6.0, 1.0/24.0,
                              1.0/120.0, 1.0/720.0, 1.0/5040.0};
    C z0 = C(0.0), z = C(0.5, 0.0);
    C val = power_series_eval(coeffs, z0, z);
    EXPECT_NEAR(val.real(), std::exp(0.5), 1e-4);
}

// ---- Blaschke product ----
TEST(CplxBlaschke, UnitModulus) {
    // |B(z)| = 1 on unit circle
    std::vector<C> zeros_bl = {C(0.3, 0.0), C(0.0, 0.4)};
    C z = C(std::cos(0.7), std::sin(0.7));  // unit circle
    C b = blaschke_product(z, zeros_bl);
    EXPECT_NEAR(std::abs(b), 1.0, 1e-10);
}

TEST(CplxMobius, FixedPoints) {
    Mobius m = mobius(C(1.0), C(1.0), C(1.0), C(0.0));  // w = (z+1)/(z)
    auto fps = m.fixed_points();
    ASSERT_EQ(fps.size(), 2u);
    for (const C& fp : fps) {
        EXPECT_NEAR(m(fp).real(), fp.real(), 1e-10);
        EXPECT_NEAR(m(fp).imag(), fp.imag(), 1e-10);
    }
}

TEST(CplxJoukowskiInv, RoundTrip) {
    C z(2.0, 1.0);
    C w = joukowski(z, 1.0);
    auto roots = joukowski_inv(w, 1.0);
    ASSERT_EQ(roots.size(), 2u);
    bool matched = false;
    for (const C& r : roots) {
        if (std::abs(r - z) < 1e-10) matched = true;
    }
    EXPECT_TRUE(matched);
}

TEST(CplxWindingNumber, OffsetCircle) {
    // Circle centred at 2+0i should wind once around 2, not around 0
    int N = 120;
    std::vector<C> gamma;
    for (int i = 0; i <= N; ++i) {
        double theta = 2.0 * M_PI * i / N;
        gamma.push_back(C(2.0 + std::cos(theta), std::sin(theta)));
    }
    EXPECT_EQ(winding_number(gamma, C(2.0, 0.0)), 1);
    EXPECT_EQ(winding_number(gamma, C(0.0, 0.0)), 0);
}

TEST(CplxContourIntegral, OneOverZ) {
    // ∮ dz/z around the unit circle = 2πi
    int N = 200;
    std::vector<C> path;
    for (int i = 0; i <= N; ++i) {
        double t = 2.0 * M_PI * i / N;
        path.push_back(C(std::cos(t), std::sin(t)));
    }
    auto f = [](C z) { return C(1.0) / z; };
    C val = contour_integral(f, path, 40);
    EXPECT_NEAR(val.real(), 0.0, 0.05);
    EXPECT_NEAR(val.imag(), 2.0 * M_PI, 0.15);
}

TEST(CplxCrossRatio, HarmonicPoints) {
    // Equally spaced real points: cross ratio is always 4/3
    C z1(-1.0), z2(0.0), z3(1.0), z4(2.0);
    EXPECT_NEAR(cross_ratio(z1, z2, z3, z4), 4.0 / 3.0, 1e-10);
}

static std::vector<C> unit_circle_contour(int n) {
    std::vector<C> path;
    for (int i = 0; i <= n; ++i) {
        double t = 2.0 * M_PI * i / n;
        path.push_back(C(std::cos(t), std::sin(t)));
    }
    return path;
}

// ---- Residue (additional) ----

TEST(CplxResidue, PoleAtI) {
    auto f = [](C z) { return C(1.0) / (z - C(0.0, 1.0)); };
    C res = residue(f, C(0.0, 1.0), 1e-5);
    EXPECT_NEAR(res.real(), 1.0, 1e-2);
    EXPECT_NEAR(res.imag(), 0.0, 1e-2);
}

TEST(CplxResidue, RationalAtOne) {
    auto f = [](C z) { return (z - C(2.0)) / (z - C(1.0)); };
    C res = residue(f, C(1.0), 1e-5);
    EXPECT_NEAR(res.real(), -1.0, 1e-2);
    EXPECT_NEAR(res.imag(), 0.0, 1e-2);
}

TEST(CplxResidue, DoublePoleZero) {
    auto f = [](C z) { return C(1.0) / ((z - C(1.0)) * (z - C(1.0))); };
    C res = residue(f, C(1.0), 1e-5);
    EXPECT_NEAR(std::abs(res), 0.0, 1e-1);
}

TEST(CplxResidue, ExpOverZ) {
    auto f = [](C z) { return std::exp(z) / z; };
    C res = residue(f, C(0.0), 1e-5);
    EXPECT_NEAR(res.real(), 1.0, 1e-2);
    EXPECT_NEAR(res.imag(), 0.0, 1e-2);
}

// ---- Cauchy integral (additional) ----

TEST(CplxCauchyIntegral, ConstantFunction) {
    auto contour = unit_circle_contour(120);
    auto f = [](C z) { (void)z; return C(3.0, -2.0); };
    C val = cauchy_integral(f, C(0.0, 0.0), contour, 60);
    EXPECT_NEAR(val.real(), 3.0, 0.15);
    EXPECT_NEAR(val.imag(), -2.0, 0.15);
}

TEST(CplxCauchyIntegral, LinearFunction) {
    auto contour = unit_circle_contour(120);
    auto f = [](C z) { return z; };
    C val = cauchy_integral(f, C(0.0, 0.0), contour, 60);
    EXPECT_NEAR(val.real(), 0.0, 0.15);
    EXPECT_NEAR(val.imag(), 0.0, 0.15);
}

TEST(CplxCauchyIntegral, OffCenterPoint) {
    auto contour = unit_circle_contour(120);
    auto f = [](C z) { return z * z; };
    C z0(0.3, 0.0);
    C val = cauchy_integral(f, z0, contour, 80);
    EXPECT_NEAR(val.real(), z0.real() * z0.real() - z0.imag() * z0.imag(), 0.2);
    EXPECT_NEAR(val.imag(), 2.0 * z0.real() * z0.imag(), 0.2);
}

TEST(CplxCauchyIntegral, QuadraticInside) {
    auto contour = unit_circle_contour(160);
    auto f = [](C z) { return z * z + C(1.0); };
    C val = cauchy_integral(f, C(0.0, 0.0), contour, 80);
    EXPECT_NEAR(val.real(), 1.0, 0.15);
    EXPECT_NEAR(val.imag(), 0.0, 0.15);
}

// ---- Möbius compose (additional) ----

TEST(CplxMobiusCompose, IdentityLeft) {
    Mobius id(C(1.0), C(0.0), C(0.0), C(1.0));
    Mobius m(C(0.0), C(1.0), C(1.0), C(0.0));
    auto comp = id.compose(m);
    C z(0.5, 0.25);
    EXPECT_NEAR(comp(z).real(), m(z).real(), 1e-10);
    EXPECT_NEAR(comp(z).imag(), m(z).imag(), 1e-10);
}

TEST(CplxMobiusCompose, IdentityRight) {
    Mobius m(C(2.0), C(1.0), C(0.0), C(1.0));
    Mobius id(C(1.0), C(0.0), C(0.0), C(1.0));
    auto comp = m.compose(id);
    C z(-0.3, 0.8);
    EXPECT_NEAR(comp(z).real(), m(z).real(), 1e-10);
    EXPECT_NEAR(comp(z).imag(), m(z).imag(), 1e-10);
}

TEST(CplxMobiusCompose, WithInverse) {
    Mobius m(C(2.0), C(1.0), C(1.0), C(1.0));
    auto comp = m.compose(m.inverse());
    C z(1.2, -0.4);
    EXPECT_NEAR(comp(z).real(), z.real(), 1e-8);
    EXPECT_NEAR(comp(z).imag(), z.imag(), 1e-8);
}

TEST(CplxMobiusCompose, TranslationThenScale) {
    Mobius shift(C(1.0), C(1.0), C(0.0), C(1.0));
    Mobius scale(C(3.0), C(0.0), C(0.0), C(1.0));
    auto comp = scale.compose(shift);
    C z(0.0, 0.0);
    C expected = scale(shift(z));
    EXPECT_NEAR(comp(z).real(), expected.real(), 1e-10);
    EXPECT_NEAR(comp(z).imag(), expected.imag(), 1e-10);
}

// ---- Joukowski (additional) ----

TEST(CplxJoukowski, UnitCircle) {
    C w = joukowski(C(1.0, 0.0), 1.0);
    EXPECT_NEAR(w.real(), 2.0, 1e-10);
    EXPECT_NEAR(w.imag(), 0.0, 1e-10);
}

TEST(CplxJoukowski, RealAxis) {
    C w = joukowski(C(2.0, 0.0), 1.0);
    EXPECT_NEAR(w.real(), 2.5, 1e-10);
    EXPECT_NEAR(w.imag(), 0.0, 1e-10);
}

TEST(CplxJoukowski, CustomC) {
    C w = joukowski(C(4.0, 0.0), 2.0);
    EXPECT_NEAR(w.real(), 5.0, 1e-10);
    EXPECT_NEAR(w.imag(), 0.0, 1e-10);
}

TEST(CplxJoukowski, Origin) {
    C w = joukowski(C(0.0, 0.0), 1.0);
    EXPECT_NEAR(w.real(), 0.0, 1e-10);
    EXPECT_NEAR(w.imag(), 0.0, 1e-10);
}

// ---- Green's function for the Dirichlet Laplacian on the unit disk ----

TEST(CplxGreenFunction, Symmetry) {
    // G(z, z0) == G(z0, z) for several interior point pairs.
    std::vector<std::pair<C, C>> pairs = {
        {C(0.3, 0.1), C(-0.2, 0.4)},
        {C(0.5, 0.0), C(0.0, 0.5)},
        {C(-0.6, -0.1), C(0.2, -0.3)},
        {C(0.1, 0.05), C(0.7, 0.2)},
    };
    for (const auto& [z, z0] : pairs) {
        double g1 = green_function_disk(z, z0);
        double g2 = green_function_disk(z0, z);
        EXPECT_NEAR(g1, g2, 1e-10);
    }
}

TEST(CplxGreenFunction, KnownValueAtOrigin) {
    // G(0, z0) = (1/(2*pi)) * ln|(0-z0)/(1-0)| = ln|z0| / (2*pi)
    C z0(0.4, 0.3);
    double expected = std::log(std::abs(z0)) / (2.0 * M_PI);
    EXPECT_NEAR(green_function_disk(C(0.0, 0.0), z0), expected, 1e-10);

    C z1(0.0, -0.6);
    double expected1 = std::log(std::abs(z1)) / (2.0 * M_PI);
    EXPECT_NEAR(green_function_disk(C(0.0, 0.0), z1), expected1, 1e-10);
}

TEST(CplxGreenFunction, KnownValueHandComputed) {
    // z = 0.5, z0 = 0.5i: |z-z0| = |0.5 - 0.5i| = 0.5*sqrt(2)
    // |1 - conj(z0)*z| = |1 - (-0.5i)*0.5| = |1 + 0.25i| = sqrt(1 + 0.0625)
    C z(0.5, 0.0), z0(0.0, 0.5);
    double num = std::abs(z - z0);
    double den = std::abs(C(1.0) - std::conj(z0) * z);
    double expected = std::log(num / den) / (2.0 * M_PI);
    EXPECT_NEAR(green_function_disk(z, z0), expected, 1e-12);
    // Sanity: matches direct hand-computed magnitudes above.
    EXPECT_NEAR(num, 0.5 * std::sqrt(2.0), 1e-12);
    EXPECT_NEAR(den, std::sqrt(1.0625), 1e-12);
}

TEST(CplxGreenFunction, NegativeInsideDisk) {
    // With this sign convention, G <= 0 everywhere inside the disk
    // (|z-z0| <= |1-conj(z0)z| for z, z0 in the closed unit disk).
    std::vector<C> pts = {C(0.1, 0.2), C(-0.3, 0.1), C(0.0, -0.5), C(0.6, 0.3)};
    C z0(0.2, -0.1);
    for (const C& z : pts) {
        if (std::abs(z - z0) < 1e-9) continue;
        EXPECT_LE(green_function_disk(z, z0), 0.0);
    }
}

TEST(CplxGreenFunction, VanishesNearBoundary) {
    // Fixed interior z0; G(z, z0) -> 0 as |z| -> 1.
    C z0(0.3, -0.2);
    double theta = 0.9;
    double prev = std::abs(green_function_disk(C(0.9 * std::cos(theta), 0.9 * std::sin(theta)), z0));
    for (double r : {0.99, 0.999, 0.9999, 0.99999}) {
        C z(r * std::cos(theta), r * std::sin(theta));
        double g = green_function_disk(z, z0);
        EXPECT_LT(std::abs(g), prev + 1e-9);  // monotonically shrinking toward 0
        prev = std::abs(g);
    }
    EXPECT_NEAR(prev, 0.0, 1e-3);
}

TEST(CplxGreenFunction, SingularityAtSource) {
    // As z -> z0, G(z, z0) -> -infinity (logarithmic singularity).
    C z0(0.2, 0.1);
    double g_far = green_function_disk(z0 + C(0.1, 0.0), z0);
    double g_near = green_function_disk(z0 + C(1e-3, 0.0), z0);
    double g_nearer = green_function_disk(z0 + C(1e-6, 0.0), z0);
    EXPECT_LT(g_near, g_far);       // grows more negative as z approaches z0
    EXPECT_LT(g_nearer, g_near);
    EXPECT_TRUE(std::isinf(green_function_disk(z0, z0)));
    EXPECT_LT(green_function_disk(z0, z0), 0.0);
}

TEST(CplxGreenFunction, PoissonKernelBoundaryDerivative) {
    // Cross-check with poisson_kernel: with this sign convention,
    // dG/dn(e^{i theta}, r e^{i phi}) = poisson_kernel(theta, phi, r) / (2*pi),
    // where n is the outward radial direction at the boundary. Since
    // G(e^{i theta}, z0) == 0 exactly, a one-sided finite difference gives
    // dG/dn ≈ -G((1-h) e^{i theta}, z0) / h.
    double r = 0.5, phi = 0.3;
    C z0(r * std::cos(phi), r * std::sin(phi));
    double h = 1e-4;
    for (double theta : {0.0, 0.7, 1.1, 2.5, 4.0, 5.5}) {
        C z_inner((1.0 - h) * std::cos(theta), (1.0 - h) * std::sin(theta));
        double deriv = -green_function_disk(z_inner, z0) / h;
        double expected = poisson_kernel(theta, phi, r) / (2.0 * M_PI);
        EXPECT_NEAR(deriv, expected, 1e-3);
    }
}

TEST(CplxGreenFunction, OutOfDomainReturnsZero) {
    // z or z0 on/outside the boundary: defensive, well-defined return of 0.0.
    C inside(0.2, 0.1);
    EXPECT_EQ(green_function_disk(C(1.0, 0.0), inside), 0.0);       // z on boundary
    EXPECT_EQ(green_function_disk(inside, C(1.0, 0.0)), 0.0);       // z0 on boundary
    EXPECT_EQ(green_function_disk(C(1.5, 0.0), inside), 0.0);       // z outside
    EXPECT_EQ(green_function_disk(inside, C(0.0, 2.0)), 0.0);       // z0 outside
    EXPECT_EQ(green_function_disk(C(3.0, 0.0), C(4.0, 0.0)), 0.0);  // both outside
}

TEST(CplxGreenFunction, GeneralRadius) {
    // On a disk of radius R, rescaling z, z0 by R should reproduce the unit-disk
    // formula: green_function_disk(R*w, R*w0, R) == green_function_disk(w, w0, 1).
    double R = 2.5;
    C w(0.3, -0.2), w0(-0.1, 0.4);
    double g_unit = green_function_disk(w, w0);
    double g_scaled = green_function_disk(R * w, R * w0, R);
    EXPECT_NEAR(g_unit, g_scaled, 1e-10);

    // Out-of-domain on the scaled disk.
    EXPECT_EQ(green_function_disk(C(R, 0.0), w0 * R, R), 0.0);
    EXPECT_EQ(green_function_disk(C(0.0, 0.0), C(0.0, 0.0), 0.0), 0.0);  // invalid radius
}
