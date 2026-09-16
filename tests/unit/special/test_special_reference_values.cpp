// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §8.4: four special functions whose VALUE nothing asserted.
//
// A sample of 24 mutants at seed 71 against the eleven suites that cover
// `src/special/special.cpp` scored 62.5%, and four survivors were of one kind:
// a function with a closed form, a mutation that changes it into a different
// function entirely, and not one test that would notice.
//
//   beta_func      lgamma(a) + lgamma(b) - lgamma(a+b) became + lgamma(a+b),
//                  so B(a,b) = Gamma(a)Gamma(b)/Gamma(a+b) became
//                  Gamma(a)Gamma(b)Gamma(a+b). Every value wrong, every test
//                  passing.
//   spherical_yn   y_n(x) = sqrt(pi/2x) Y_{n+1/2}(x) became Y_{n-1/2}(x) --
//                  a different order of a different function. The j
//                  counterpart on the line above it IS asserted, which is why
//                  that one's mutant dies.
//   legendre_p     the domain guard `x > 1.0` became `x >= 1.0`, so P_n(1)
//                  becomes NaN. P_n(1) = 1 is the first value in the table.
//   erfinv         erfinv(-1) returned +infinity instead of -infinity.
//
// What these have in common is that the suites around them assert SHAPE --
// finiteness, sign, monotonicity, a recurrence relating one call to another --
// and a recurrence is satisfied by a whole family of functions, not only the
// right one. These compare against closed forms written out here, which is the
// one comparison that cannot be satisfied by a different function.

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/special/special.hpp"

using namespace ms;

TEST(SpecialReferenceValues, BetaFunctionMatchesItsDefinition) {
    // B(a, b) = Gamma(a)Gamma(b)/Gamma(a+b). For positive integers that is
    // (a-1)!(b-1)!/(a+b-1)!, which is exact in double at these sizes.
    EXPECT_NEAR(beta_func(1.0, 1.0), 1.0, 1e-14);
    EXPECT_NEAR(beta_func(1.0, 2.0), 0.5, 1e-14);
    EXPECT_NEAR(beta_func(2.0, 1.0), 0.5, 1e-14);
    EXPECT_NEAR(beta_func(2.0, 3.0), 1.0 / 12.0, 1e-14);      // 1!*2!/4! = 2/24
    EXPECT_NEAR(beta_func(3.0, 4.0), 1.0 / 60.0, 1e-15);      // 2!*3!/6! = 12/720
    EXPECT_NEAR(beta_func(5.0, 5.0), 1.0 / 630.0, 1e-16);     // 4!*4!/9!
    EXPECT_NEAR(beta_func(4.0, 2.0), 1.0 / 20.0, 1e-15);

    // B(1/2, 1/2) = pi, the one every table opens with.
    EXPECT_NEAR(beta_func(0.5, 0.5), M_PI, 1e-13);
    // B(1/2, 1) = 2, and B(3/2, 1/2) = pi/2.
    EXPECT_NEAR(beta_func(0.5, 1.0), 2.0, 1e-14);
    EXPECT_NEAR(beta_func(1.5, 0.5), M_PI / 2.0, 1e-13);

    // Symmetry, and the recurrence B(a+1, b) = B(a, b) * a/(a+b) -- true of the
    // real thing and false of the product form, since the latter's Gamma(a+b)
    // moves the wrong way.
    for (const double a : {0.75, 1.5, 2.25, 3.5}) {
        for (const double b : {0.5, 1.25, 2.0, 4.5}) {
            EXPECT_NEAR(beta_func(a, b), beta_func(b, a), 1e-12) << a << ", " << b;
            EXPECT_NEAR(beta_func(a + 1.0, b), beta_func(a, b) * a / (a + b), 1e-12)
                << a << ", " << b;
        }
    }
}

TEST(SpecialReferenceValues, SphericalBesselMatchesItsClosedForms) {
    // j_0, j_1, j_2 and y_0, y_1, y_2 have elementary closed forms. They are
    // written out here rather than taken from a recurrence, because a recurrence
    // is what the implementation already is.
    for (const double x : {0.3, 0.7, 1.0, 2.5, 4.0, 7.5, 13.0}) {
        const double s = std::sin(x), c = std::cos(x);
        const double j0 = s / x;
        const double j1 = s / (x * x) - c / x;
        const double j2 = (3.0 / (x * x * x) - 1.0 / x) * s - (3.0 / (x * x)) * c;
        const double j3 = (15.0 / (x * x * x * x) - 6.0 / (x * x)) * s
                        - (15.0 / (x * x * x) - 1.0 / x) * c;
        const double y0 = -c / x;
        const double y1 = -c / (x * x) - s / x;
        const double y2 = (-3.0 / (x * x * x) + 1.0 / x) * c - (3.0 / (x * x)) * s;
        const double y3 = (-15.0 / (x * x * x * x) + 6.0 / (x * x)) * c
                        - (15.0 / (x * x * x) - 1.0 / x) * s;

        // Two tolerances, because the two families are not computed to the same
        // accuracy and saying so is part of the contract. `spherical_jn` reaches
        // machine precision; `spherical_yn` goes through `bessel_y_general` at
        // half-integer order, whose series is good to about EIGHT significant
        // figures -- measured, the worst relative disagreement over these seven
        // arguments and four orders is 1.9e-8. That is the accuracy the function
        // has, and a test asserting 1e-10 would be asserting an accuracy it does
        // not claim. It is still four orders of magnitude tighter than anything
        // that could confuse y_n with a neighbouring order.
        const double jtol = 1e-12 * std::max(1.0, std::abs(j3));
        const double ytol = 1e-6 * std::max(1.0, std::abs(y3));
        EXPECT_NEAR(spherical_jn(0, x), j0, jtol) << "j0 at " << x;
        EXPECT_NEAR(spherical_jn(1, x), j1, jtol) << "j1 at " << x;
        EXPECT_NEAR(spherical_jn(2, x), j2, jtol) << "j2 at " << x;
        EXPECT_NEAR(spherical_jn(3, x), j3, jtol) << "j3 at " << x;
        EXPECT_NEAR(spherical_yn(0, x), y0, ytol) << "y0 at " << x;
        EXPECT_NEAR(spherical_yn(1, x), y1, ytol) << "y1 at " << x;
        EXPECT_NEAR(spherical_yn(2, x), y2, ytol) << "y2 at " << x;
        EXPECT_NEAR(spherical_yn(3, x), y3, ytol) << "y3 at " << x;

        // The Wronskian, which ties the two families together and which no
        // single-family recurrence can satisfy by accident:
        // j_n(x) y_n'(x) - j_n'(x) y_n(x) = 1/x^2, and in the form used here
        // j_{n+1}(x) y_n(x) - j_n(x) y_{n+1}(x) = 1/x^2.
        for (int n = 0; n <= 2; ++n) {
            const double w = spherical_jn(n + 1, x) * spherical_yn(n, x)
                           - spherical_jn(n, x) * spherical_yn(n + 1, x);
            EXPECT_NEAR(w, 1.0 / (x * x), 1e-6 * std::max(1.0, 1.0 / (x * x)))
                << "Wronskian at n=" << n << ", x=" << x;
        }
    }

    // Out of domain, both ways.
    EXPECT_TRUE(std::isnan(spherical_yn(-1, 1.0)));
    EXPECT_TRUE(std::isnan(spherical_yn(1, 0.0)));
    EXPECT_TRUE(std::isnan(spherical_yn(1, -1.0)));
}

TEST(SpecialReferenceValues, LegendreAtTheEndpointsOfItsDomain) {
    // P_n(1) = 1 and P_n(-1) = (-1)^n -- the first two rows of every table, and
    // the values a domain guard written with the wrong comparison turns into
    // NaN. The interval is closed: +/-1 are in the domain, not outside it.
    for (int n = 0; n <= 8; ++n) {
        EXPECT_NEAR(legendre_p(n, 1.0), 1.0, 1e-12) << "P_" << n << "(1)";
        EXPECT_NEAR(legendre_p(n, -1.0), (n % 2 == 0) ? 1.0 : -1.0, 1e-12)
            << "P_" << n << "(-1)";
    }
    // Just outside it is NaN, which is what makes the above a boundary rather
    // than a bound.
    EXPECT_TRUE(std::isnan(legendre_p(2, 1.0 + 1e-9)));
    EXPECT_TRUE(std::isnan(legendre_p(2, -1.0 - 1e-9)));
    EXPECT_TRUE(std::isnan(legendre_p(-1, 0.5)));

    // And the interior values against their closed forms, so the endpoint
    // assertions cannot be satisfied by a function that is 1 everywhere.
    for (const double x : {-0.9, -0.4, 0.0, 0.25, 0.8}) {
        EXPECT_NEAR(legendre_p(0, x), 1.0, 1e-13);
        EXPECT_NEAR(legendre_p(1, x), x, 1e-13);
        EXPECT_NEAR(legendre_p(2, x), 0.5 * (3.0 * x * x - 1.0), 1e-13);
        EXPECT_NEAR(legendre_p(3, x), 0.5 * (5.0 * x * x * x - 3.0 * x), 1e-13);
        EXPECT_NEAR(legendre_p(4, x),
                    0.125 * (35.0 * x * x * x * x - 30.0 * x * x + 3.0), 1e-13);
    }
}

TEST(SpecialReferenceValues, ErfInverseAtTheEndsOfItsRange) {
    // erf maps the reals onto (-1, 1), so its inverse is infinite at both ends
    // -- and the sign of the infinity is the whole content of the statement.
    EXPECT_TRUE(std::isinf(erfinv(1.0)));
    EXPECT_GT(erfinv(1.0), 0.0);
    EXPECT_TRUE(std::isinf(erfinv(-1.0)));
    EXPECT_LT(erfinv(-1.0), 0.0) << "erfinv(-1) is not negative";
    EXPECT_EQ(erfinv(0.0), 0.0);

    // Odd symmetry and a real round trip on either side of zero.
    for (const double p : {0.05, 0.2, 0.5, 0.75, 0.95, 0.999}) {
        EXPECT_NEAR(erfinv(-p), -erfinv(p), 1e-12) << p;
        EXPECT_NEAR(ms::erf(erfinv(p)), p, 1e-10) << p;
        EXPECT_NEAR(ms::erf(erfinv(-p)), -p, 1e-10) << p;
    }
}

TEST(SpecialReferenceValues, ParabolicCylinderWIsTheCombinationDlmfSpecifies) {
    // `pcf_w` has already been wrong once -- the comment above it records that a
    // `U cos(pi a) - V sin(pi a)` mix was a different function -- and the sign in
    // the corrected DLMF 12.14.4 combination was still unasserted.
    //
    // The differential equation cannot see it: w1 and w2 both solve
    // y'' + (x^2/4 - a) y = 0, so EVERY linear combination of them does too, and a
    // residual test passes for the wrong one. What identifies this particular
    // combination is its behaviour at the origin, where w1 is 1 with slope 0 and
    // w2 is 0 with slope 1, so W(a,0) and W'(a,0) read the two coefficients
    // straight off -- including the minus sign between them.
    const double g1 = std::tgamma(0.25);
    const double g3 = std::tgamma(0.75);
    EXPECT_NEAR(pcf_w(0.0, 0.0), std::pow(2.0, -0.75) * std::sqrt(g1 / g3), 1e-10);

    const double h = 1e-5;
    const double slope_at_zero = (pcf_w(0.0, h) - pcf_w(0.0, -h)) / (2.0 * h);
    EXPECT_NEAR(slope_at_zero, -std::pow(2.0, -0.25) * std::sqrt(g3 / g1), 1e-8);
    EXPECT_LT(slope_at_zero, 0.0) << "the second term entered with the wrong sign";

    for (const double a : {-2.0, -0.5, 0.0, 0.5, 1.0, 2.0}) {
        const double slope = (pcf_w(a, h) - pcf_w(a, -h)) / (2.0 * h);
        EXPECT_LT(slope, 0.0) << "W'(a, 0) at a = " << a;
        EXPECT_GT(pcf_w(a, 0.0), 0.0) << "W(a, 0) at a = " << a;
        // |Gamma(1/4 + ia/2)| is even in a, so both coefficients are, and so is
        // W(a, 0). A combination built from the wrong moduli is not.
        EXPECT_NEAR(pcf_w(a, 0.0), pcf_w(-a, 0.0), 1e-12) << "W(a,0) not even at " << a;
    }

    // And it does solve the equation, which the above does not imply on its own.
    for (const double a : {-1.0, 0.0, 1.5}) {
        for (const double x : {-1.5, -0.5, 0.5, 1.5}) {
            const double k = 1e-4;
            const double second = (pcf_w(a, x + k) - 2.0 * pcf_w(a, x) + pcf_w(a, x - k)) / (k * k);
            const double residual = second + (x * x / 4.0 - a) * pcf_w(a, x);
            EXPECT_NEAR(residual, 0.0, 1e-4) << "a=" << a << " x=" << x;
        }
    }
}
