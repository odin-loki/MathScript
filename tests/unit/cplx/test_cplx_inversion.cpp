// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#define _USE_MATH_DEFINES
#include "ms/cplx/cplx.hpp"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <limits>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms::cplx;

namespace {
const C kOrigin(0.0, 0.0);
}  // namespace

// ---- Circle inversion: the unit circle at the origin ----
TEST(CplxInversionMap, UnitCircleSwapsInsideOutside) {
    // sigma(z) = 1/conj(z) for the unit circle centred at the origin.
    C w = apply_inversion(C(2.0, 0.0), kOrigin, 1.0);
    EXPECT_NEAR(w.real(), 0.5, 1e-15);
    EXPECT_NEAR(w.imag(), 0.0, 1e-15);

    w = apply_inversion(C(0.5, 0.0), kOrigin, 1.0);
    EXPECT_NEAR(w.real(), 2.0, 1e-15);
    EXPECT_NEAR(w.imag(), 0.0, 1e-15);

    w = apply_inversion(C(0.25, 0.0), kOrigin, 1.0);
    EXPECT_NEAR(w.real(), 4.0, 1e-15);
    EXPECT_NEAR(w.imag(), 0.0, 1e-15);

    w = apply_inversion(C(4.0, 0.0), kOrigin, 1.0);
    EXPECT_NEAR(w.real(), 0.25, 1e-15);
    EXPECT_NEAR(w.imag(), 0.0, 1e-15);

    // i is on the circle, so it is fixed.
    w = apply_inversion(C(0.0, 1.0), kOrigin, 1.0);
    EXPECT_NEAR(w.real(), 0.0, 1e-15);
    EXPECT_NEAR(w.imag(), 1.0, 1e-15);

    // 0.1 + 0.2i has |z|^2 = 0.05, so 1/conj(z) = z/|z|^2 = 2 + 4i.
    w = apply_inversion(C(0.1, 0.2), kOrigin, 1.0);
    EXPECT_NEAR(w.real(), 2.0, 1e-15);
    EXPECT_NEAR(w.imag(), 4.0, 1e-15);

    EXPECT_GT(std::abs(apply_inversion(C(0.25, 0.0), kOrigin, 1.0)), 1.0);
    EXPECT_LT(std::abs(apply_inversion(C(4.0, 0.0), kOrigin, 1.0)), 1.0);
}

TEST(CplxInversionMap, UnitCircleFixesEveryBoundaryPoint) {
    for (int k = 0; k < 16; ++k) {
        const double t = 2.0 * M_PI * k / 16.0;
        const C z(std::cos(t), std::sin(t));
        EXPECT_NEAR(std::abs(apply_inversion(z, kOrigin, 1.0) - z), 0.0, 1e-15);
    }
}

TEST(CplxInversionMap, UnitCircleIsOneOverConjugate) {
    const C zs[] = {C(0.3, -1.7), C(2.0, 5.0), C(-0.6, 0.1)};
    for (const C& z : zs) {
        const C w = apply_inversion(z, kOrigin, 1.0);
        EXPECT_NEAR(std::abs(w - C(1.0, 0.0) / std::conj(z)), 0.0, 1e-15);
    }
}

// ---- Circle inversion: a general circle ----
TEST(CplxInversionMap, GeneralCircleFixesItsOwnBoundary) {
    const C centre(1.0, 1.0);
    const double r = 2.0;

    C w = apply_inversion(C(3.0, 1.0), centre, r);
    EXPECT_NEAR(w.real(), 3.0, 1e-15);
    EXPECT_NEAR(w.imag(), 1.0, 1e-15);

    w = apply_inversion(C(1.0, 3.0), centre, r);
    EXPECT_NEAR(w.real(), 1.0, 1e-15);
    EXPECT_NEAR(w.imag(), 3.0, 1e-15);

    for (int k = 0; k < 12; ++k) {
        const double t = 2.0 * M_PI * k / 12.0;
        const C z(1.0 + 2.0 * std::cos(t), 1.0 + 2.0 * std::sin(t));
        EXPECT_NEAR(std::abs(apply_inversion(z, centre, r) - z), 0.0, 1e-14);
    }
}

TEST(CplxInversionMap, HandComputedValues) {
    // z = 2+3i, centre 1+i, r = 2:  1+i + 4/(1-2i) = 1+i + 0.8+1.6i = 1.8+2.6i
    C w = apply_inversion(C(2.0, 3.0), C(1.0, 1.0), 2.0);
    EXPECT_NEAR(w.real(), 1.8, 1e-14);
    EXPECT_NEAR(w.imag(), 2.6, 1e-14);

    // z = 0.5-0.25i, centre 1+2i, r = 3:  13/85 - (154/85) i
    w = apply_inversion(C(0.5, -0.25), C(1.0, 2.0), 3.0);
    EXPECT_NEAR(w.real(), 13.0 / 85.0, 1e-12);
    EXPECT_NEAR(w.imag(), -154.0 / 85.0, 1e-12);
}

TEST(CplxInversionMap, InsideMapsOutsideWithProductR2) {
    const C centre(1.0, 1.0);
    const double r = 2.0;
    const C z(1.5, 1.0);  // distance 0.5 from the centre, strictly inside
    const C w = apply_inversion(z, centre, r);
    EXPECT_NEAR(w.real(), 9.0, 1e-15);
    EXPECT_NEAR(w.imag(), 1.0, 1e-15);
    EXPECT_NEAR(std::abs(z - centre) * std::abs(w - centre), 4.0, 1e-12);
    EXPECT_LT(std::abs(z - centre), r);
    EXPECT_GT(std::abs(w - centre), r);
}

TEST(CplxInversionMap, IsAnInvolution) {
    struct Row {
        C z;
        C centre;
        double r;
        C expected;
    };
    const Row rows[] = {
        {C(0.1, 0.2), C(0.0, 0.0), 1.0, C(2.0, 4.0)},
        {C(-3.0, 4.0), C(2.0, -1.0), 0.5, C(1.975, -0.975)},
        {C(7.0, 0.0), C(7.0, 3.0), 2.25, C(7.0, 1.3125)},
        {C(2.5, -0.75), C(-1.0, 0.5), 1.75,
         C(-0.22398190045248878, 0.22285067873303172)},
    };
    for (const Row& row : rows) {
        const C w = apply_inversion(row.z, row.centre, row.r);
        EXPECT_NEAR(std::abs(w - row.expected), 0.0, 1e-12);
        const C back = apply_inversion(w, row.centre, row.r);
        EXPECT_NEAR(std::abs(back - row.z), 0.0, 1e-12);
    }
}

TEST(CplxInversionMap, RayAndRadiusInvariants) {
    const C z(2.5, -0.75);
    const C centre(-1.0, 0.5);
    const double r = 1.75;
    const C w = apply_inversion(z, centre, r);
    // |z-c| * |w-c| == r^2
    EXPECT_NEAR(std::abs(z - centre) * std::abs(w - centre), r * r, 1e-12);
    // w - c lies on the same ray out of the centre as z - c: the ratio is a
    // positive real, equal to r^2 / |z-c|^2.
    const C q = (w - centre) / (z - centre);
    EXPECT_NEAR(q.imag(), 0.0, 1e-15);
    EXPECT_GT(q.real(), 0.0);
    EXPECT_NEAR(q.real(), r * r / std::norm(z - centre), 1e-12);
}

// ---- The Mobius-coefficient form ----
TEST(CplxInversionMap, MobiusCoefficientsAndDeterminant) {
    const Mobius m = inversion(C(1.0, 1.0), 2.0);
    EXPECT_NEAR(m.a.real(), 1.0, 1e-15);
    EXPECT_NEAR(m.a.imag(), 1.0, 1e-15);
    EXPECT_NEAR(m.b.real(), 2.0, 1e-15);  // r^2 - |c|^2 = 4 - 2
    EXPECT_NEAR(m.b.imag(), 0.0, 1e-15);
    EXPECT_NEAR(m.c.real(), 1.0, 1e-15);
    EXPECT_NEAR(m.c.imag(), 0.0, 1e-15);
    EXPECT_NEAR(m.d.real(), -1.0, 1e-15);
    EXPECT_NEAR(m.d.imag(), 1.0, 1e-15);

    const C det = m.a * m.d - m.b * m.c;
    EXPECT_NEAR(det.real(), -4.0, 1e-15);  // det == -r^2
    EXPECT_NEAR(det.imag(), 0.0, 1e-15);

    // The unit circle at the origin is u -> 1/u, i.e. z -> 1/conj(z).
    const Mobius unit = inversion(kOrigin, 1.0);
    EXPECT_NEAR(unit.a.real(), 0.0, 1e-15);
    EXPECT_NEAR(unit.a.imag(), 0.0, 1e-15);
    EXPECT_NEAR(unit.b.real(), 1.0, 1e-15);
    EXPECT_NEAR(unit.b.imag(), 0.0, 1e-15);
    EXPECT_NEAR(unit.c.real(), 1.0, 1e-15);
    EXPECT_NEAR(unit.c.imag(), 0.0, 1e-15);
    EXPECT_NEAR(unit.d.real(), 0.0, 1e-15);
    EXPECT_NEAR(unit.d.imag(), 0.0, 1e-15);
}

TEST(CplxInversionMap, IsNotTheIdentityMobius) {
    // Regression guard: inversion() used to ignore both arguments and return
    // the identity quadruple {1, 0, 0, 1}.
    const Mobius m = inversion(C(1.0, 2.0), 3.0);
    EXPECT_NEAR(m.a.real(), 1.0, 1e-15);
    EXPECT_NEAR(m.a.imag(), 2.0, 1e-15);
    EXPECT_NEAR(m.b.real(), 4.0, 1e-15);  // r^2 - |c|^2 = 9 - 5
    EXPECT_NEAR(m.b.imag(), 0.0, 1e-15);
    EXPECT_NEAR(m.c.real(), 1.0, 1e-15);
    EXPECT_NEAR(m.c.imag(), 0.0, 1e-15);
    EXPECT_NEAR(m.d.real(), -1.0, 1e-15);
    EXPECT_NEAR(m.d.imag(), 2.0, 1e-15);
    // Applied to conj(z) it is the true inversion: 13/85 - (154/85) i.
    const C z(0.5, -0.25);
    const C w = m(std::conj(z));
    EXPECT_NEAR(w.real(), 13.0 / 85.0, 1e-12);
    EXPECT_NEAR(w.imag(), -154.0 / 85.0, 1e-12);
    // ...and not the identity's answer.
    EXPECT_GT(std::abs(w - z), 1.0);
}

TEST(CplxInversionMap, MobiusFormMatchesDirectForm) {
    const C centre(-0.4, 1.3);
    const double r = 2.5;
    const Mobius m = inversion(centre, r);
    const C zs[] = {C(0.7, -1.9), C(3.0, 0.0), C(-2.0, -2.0), C(0.5, 1.4),
                    C(10.0, 10.0)};
    for (const C& z : zs) {
        EXPECT_NEAR(std::abs(m(std::conj(z)) - apply_inversion(z, centre, r)),
                    0.0, 1e-12);
    }
}

TEST(CplxInversionMap, MobiusInverseIsTheConjugateMap) {
    // sigma(z) = M(conj(z)) is an involution, which is equivalent to the
    // coefficient identity conj(M^-1(w)) == M(conj(w)) for all w.
    const C centre(-0.4, 1.3);
    const double r = 2.5;
    const C w(0.7, -1.9);
    const Mobius m = inversion(centre, r);
    const C lhs = std::conj(m.inverse()(w));
    const C rhs = m(std::conj(w));
    EXPECT_NEAR(lhs.real(), rhs.real(), 1e-12);
    EXPECT_NEAR(lhs.imag(), rhs.imag(), 1e-12);
    EXPECT_NEAR(lhs.real(), 0.20043668122270747, 1e-12);
    EXPECT_NEAR(lhs.imag(), -0.44672489082969458, 1e-12);
}

// ---- Documented degenerate conventions ----
TEST(CplxInversionMap, CentreMapsToInfinityAndBack) {
    const C centre(1.0, 1.0);
    const C w = apply_inversion(centre, centre, 2.0);
    EXPECT_TRUE(std::isinf(w.real()));
    EXPECT_TRUE(std::isinf(w.imag()));
    EXPECT_TRUE(std::isinf(std::abs(w)));
    // The point at infinity maps back to the centre, so the involution is total.
    const C back = apply_inversion(w, centre, 2.0);
    EXPECT_NEAR(back.real(), 1.0, 1e-15);
    EXPECT_NEAR(back.imag(), 1.0, 1e-15);
}

TEST(CplxInversionMap, InfiniteInputMapsToCentre) {
    const double inf = std::numeric_limits<double>::infinity();
    const C centre(-2.0, 0.5);
    const C from_both = apply_inversion(C(inf, inf), centre, 3.0);
    EXPECT_NEAR(from_both.real(), -2.0, 1e-15);
    EXPECT_NEAR(from_both.imag(), 0.5, 1e-15);
    const C from_one = apply_inversion(C(1.0, -inf), centre, 3.0);
    EXPECT_NEAR(from_one.real(), -2.0, 1e-15);
    EXPECT_NEAR(from_one.imag(), 0.5, 1e-15);
}

TEST(CplxInversionMap, NanInputPropagatesNan) {
    const double nan_value = std::numeric_limits<double>::quiet_NaN();
    const C w = apply_inversion(C(nan_value, 0.0), C(1.0, 1.0), 2.0);
    EXPECT_TRUE(std::isnan(w.real()) || std::isnan(w.imag()));
}

TEST(CplxInversionMap, ZeroRadiusCollapsesToCentre) {
    const C centre(1.0, 1.0);
    const C zs[] = {C(2.0, 3.0), C(0.0, 0.0), C(1.0, 1.0), C(-5.0, -5.0)};
    for (const C& z : zs) {
        const C w = apply_inversion(z, centre, 0.0);
        EXPECT_NEAR(w.real(), 1.0, 1e-15);
        EXPECT_NEAR(w.imag(), 1.0, 1e-15);
        EXPECT_FALSE(std::isnan(w.real()));
        EXPECT_FALSE(std::isnan(w.imag()));
    }
    // The Mobius form degenerates to determinant zero at r == 0.
    const Mobius m = inversion(centre, 0.0);
    const C det = m.a * m.d - m.b * m.c;
    EXPECT_NEAR(det.real(), 0.0, 1e-15);
    EXPECT_NEAR(det.imag(), 0.0, 1e-15);
}

TEST(CplxInversionMap, NegativeRadiusIsTheSameCircle) {
    const C z(2.0, 3.0);
    const C centre(1.0, 1.0);
    const C neg = apply_inversion(z, centre, -2.0);
    const C pos = apply_inversion(z, centre, 2.0);
    EXPECT_NEAR(neg.real(), 1.8, 1e-14);
    EXPECT_NEAR(neg.imag(), 2.6, 1e-14);
    EXPECT_EQ(neg.real(), pos.real());
    EXPECT_EQ(neg.imag(), pos.imag());
    EXPECT_NEAR(inversion(centre, -2.0).b.real(), 2.0, 1e-15);
}

TEST(CplxInversionMap, CircleReflectIsAnAlias) {
    const C z(1.3, -2.6);
    const C centre(0.5, 0.5);
    const double r = 1.5;
    EXPECT_EQ(circle_reflect(z, centre, r).real(),
              apply_inversion(z, centre, r).real());
    EXPECT_EQ(circle_reflect(z, centre, r).imag(),
              apply_inversion(z, centre, r).imag());
    EXPECT_EQ(circle_reflect(centre, centre, 0.0).real(), centre.real());
}

// ---- Complex cross ratio ----
TEST(CplxCrossRatioComplex, UnitCirclePlusOriginIsOneMinusI) {
    // ((0-i)(1+1)) / ((0+1)(1-i)) = (-2i)/(1-i) = 1 - i
    const C cr = cross_ratio_c(C(0.0, 0.0), C(1.0, 0.0), C(0.0, 1.0),
                               C(-1.0, 0.0));
    EXPECT_NEAR(cr.real(), 1.0, 1e-15);
    EXPECT_NEAR(cr.imag(), -1.0, 1e-15);
    // The double overload keeps only the real part.
    EXPECT_NEAR(cross_ratio(C(0.0, 0.0), C(1.0, 0.0), C(0.0, 1.0),
                            C(-1.0, 0.0)),
                1.0, 1e-15);
}

TEST(CplxCrossRatioComplex, FourthRootsOfUnityIsExactlyTwo) {
    // ((1+1)(i+i)) / ((1+i)(i+1)) = 4i / 2i = 2
    const C cr = cross_ratio_c(C(1.0, 0.0), C(0.0, 1.0), C(-1.0, 0.0),
                               C(0.0, -1.0));
    EXPECT_NEAR(cr.real(), 2.0, 1e-15);
    EXPECT_NEAR(cr.imag(), 0.0, 1e-15);
    EXPECT_NEAR(cross_ratio(C(1.0, 0.0), C(0.0, 1.0), C(-1.0, 0.0),
                            C(0.0, -1.0)),
                2.0, 1e-15);
}

TEST(CplxCrossRatioComplex, CollinearIsRealAndMatchesDoubleOverload) {
    struct Row {
        C z1, z2, z3, z4;
    };
    const Row rows[] = {
        {C(0.0, 0.0), C(1.0, 0.0), C(2.0, 0.0), C(3.0, 0.0)},
        {C(-1.0, 0.0), C(0.0, 0.0), C(1.0, 0.0), C(2.0, 0.0)},
        // Collinear but not on the real axis: direction 1 + 2i.
        {C(1.0, 1.0), C(2.0, 3.0), C(3.0, 5.0), C(4.0, 7.0)},
    };
    for (const Row& row : rows) {
        const C cr = cross_ratio_c(row.z1, row.z2, row.z3, row.z4);
        EXPECT_NEAR(cr.real(), 4.0 / 3.0, 1e-12);
        EXPECT_NEAR(cr.imag(), 0.0, 1e-15);
        EXPECT_EQ(cross_ratio(row.z1, row.z2, row.z3, row.z4), cr.real());
    }
}

TEST(CplxCrossRatioComplex, ConcyclicIsReal) {
    const double t = 3.0 * M_PI / 4.0;
    const C cr = cross_ratio_c(C(1.0, 0.0), C(0.0, 1.0), C(-1.0, 0.0),
                               C(std::cos(t), std::sin(t)));
    EXPECT_NEAR(cr.real(), 2.0 - std::sqrt(2.0), 1e-12);
    EXPECT_NEAR(cr.imag(), 0.0, 1e-12);

    // A general circle: centre -2 + i, radius 3, at four arbitrary angles.
    const C centre(-2.0, 1.0);
    const double radius = 3.0;
    const double angles[] = {0.3, 1.1, 2.9, 5.0};
    C p[4] = {};
    for (int i = 0; i < 4; ++i) {
        p[i] = centre + C(radius * std::cos(angles[i]),
                          radius * std::sin(angles[i]));
    }
    const C cr2 = cross_ratio_c(p[0], p[1], p[2], p[3]);
    EXPECT_NEAR(cr2.imag(), 0.0, 1e-12);
    EXPECT_EQ(cross_ratio(p[0], p[1], p[2], p[3]), cr2.real());
}

TEST(CplxCrossRatioComplex, GeneralPositionHasNonZeroImaginaryPart) {
    // ((2-1)(0-i)) / ((2-i)(0-1)) = (-i)/(-2+i) = (2i-1)/5 = -0.2 + 0.4i
    const C cr = cross_ratio_c(C(2.0, 0.0), C(0.0, 0.0), C(1.0, 0.0),
                               C(0.0, 1.0));
    EXPECT_NEAR(cr.real(), -0.2, 1e-15);
    EXPECT_NEAR(cr.imag(), 0.4, 1e-15);
    EXPECT_GT(std::abs(cr.imag()), 0.1);
    // The double overload is lossy here: it reports only -0.2.
    EXPECT_NEAR(cross_ratio(C(2.0, 0.0), C(0.0, 0.0), C(1.0, 0.0),
                            C(0.0, 1.0)),
                -0.2, 1e-15);
}

TEST(CplxCrossRatioComplex, InvariantUnderMobius) {
    const C p1(0.0, 0.0), p2(1.0, 0.0), p3(0.0, 1.0), p4(-1.0, 0.0);
    const C base = cross_ratio_c(p1, p2, p3, p4);  // 1 - i

    const Mobius m1(C(2.0, 0.0), C(1.0, 0.0), C(1.0, 0.0), C(3.0, 0.0));
    const C v1 = cross_ratio_c(m1(p1), m1(p2), m1(p3), m1(p4));
    EXPECT_NEAR(v1.real(), base.real(), 1e-12);
    EXPECT_NEAR(v1.imag(), base.imag(), 1e-12);

    const Mobius m2(C(0.0, 1.0), C(2.0, -1.0), C(1.0, 1.0), C(-1.0, 2.0));
    const C v2 = cross_ratio_c(m2(p1), m2(p2), m2(p3), m2(p4));
    EXPECT_NEAR(v2.real(), base.real(), 1e-12);
    EXPECT_NEAR(v2.imag(), base.imag(), 1e-12);
}

TEST(CplxCrossRatioComplex, CoincidentDenominatorIsInfinity) {
    const C a = cross_ratio_c(C(0.0, 0.0), C(1.0, 0.0), C(2.0, 0.0),
                              C(0.0, 0.0));  // z1 == z4
    EXPECT_TRUE(std::isinf(a.real()));
    EXPECT_TRUE(std::isinf(a.imag()));
    EXPECT_TRUE(std::isinf(std::abs(a)));
    EXPECT_TRUE(std::isinf(cross_ratio(C(0.0, 0.0), C(1.0, 0.0), C(2.0, 0.0),
                                       C(0.0, 0.0))));

    const C b = cross_ratio_c(C(0.0, 0.0), C(1.0, 0.0), C(1.0, 0.0),
                              C(2.0, 0.0));  // z2 == z3
    EXPECT_TRUE(std::isinf(b.real()));
    EXPECT_TRUE(std::isinf(b.imag()));
}

TEST(CplxCrossRatioComplex, TotallyDegenerateIsNaN) {
    // z1 == z3 == z4 == 0: numerator and denominator are both exactly zero.
    const C cr = cross_ratio_c(C(0.0, 0.0), C(1.0, 0.0), C(0.0, 0.0),
                               C(0.0, 0.0));
    EXPECT_TRUE(std::isnan(cr.real()));
    EXPECT_TRUE(std::isnan(cr.imag()));
    EXPECT_TRUE(std::isnan(cross_ratio(C(0.0, 0.0), C(1.0, 0.0), C(0.0, 0.0),
                                       C(0.0, 0.0))));
}

TEST(CplxCrossRatioComplex, NumeratorZeroAndEqualFirstPair) {
    // z1 == z3 with a non-zero denominator: the cross ratio is exactly 0.
    const C zero = cross_ratio_c(C(0.0, 0.0), C(1.0, 0.0), C(0.0, 0.0),
                                 C(3.0, 0.0));
    EXPECT_NEAR(zero.real(), 0.0, 1e-15);
    EXPECT_NEAR(zero.imag(), 0.0, 1e-15);
    EXPECT_EQ(cross_ratio(C(0.0, 0.0), C(1.0, 0.0), C(0.0, 0.0), C(3.0, 0.0)),
              0.0);

    // z1 == z2: ((5-1)(5-2)) / ((5-2)(5-1)) = 12/12 = 1.
    const C one = cross_ratio_c(C(5.0, 0.0), C(5.0, 0.0), C(1.0, 0.0),
                                C(2.0, 0.0));
    EXPECT_NEAR(one.real(), 1.0, 1e-15);
    EXPECT_NEAR(one.imag(), 0.0, 1e-15);
}

// ---- Cross ratio and inversion together ----
TEST(CplxCrossRatioComplex, InversionConjugatesTheCrossRatio) {
    // Circle inversion is anti-Mobius, so it conjugates the cross ratio.
    const C p1(0.4, 0.1), p2(-1.2, 0.7), p3(2.0, -0.5), p4(0.3, 1.8);
    const C centre(0.25, -0.5);
    const double r = 1.4;
    const C base = cross_ratio_c(p1, p2, p3, p4);
    const C mapped = cross_ratio_c(apply_inversion(p1, centre, r),
                                   apply_inversion(p2, centre, r),
                                   apply_inversion(p3, centre, r),
                                   apply_inversion(p4, centre, r));
    EXPECT_NEAR(mapped.real(), base.real(), 1e-12);
    EXPECT_NEAR(mapped.imag(), -base.imag(), 1e-12);
    // The four points are in general position, so this is a real constraint.
    EXPECT_GT(std::abs(base.imag()), 1e-3);
}
