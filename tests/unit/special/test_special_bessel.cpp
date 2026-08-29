#include <gtest/gtest.h>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "ms/special/special.hpp"

using namespace ms;

TEST(SpecialBesselTest, bessel_y_and_k) {
    EXPECT_NEAR(bessel_y0(1.0), 0.088256964215677, 1e-6);
    EXPECT_NEAR(bessel_y1(1.0), -0.7812128213002889, 1e-6);
    EXPECT_NEAR(bessel_i(0, 1.0), 1.2660658777520084, 1e-6);
    EXPECT_NEAR(bessel_k(0, 1.0), 0.42102443824070834, 1e-6);
    EXPECT_NEAR(bessel_k(1, 1.0), 0.6019072301972346, 1e-3);
}

TEST(SpecialBesselTest, spherical_bessel) {
    EXPECT_NEAR(spherical_jn(0, 1.0), 0.8414709848078965, 1e-6);
    EXPECT_NEAR(spherical_jn(1, 1.0), 0.3011686789397571, 1e-6);
    EXPECT_NEAR(spherical_jn(2, 1.0), 0.062035052011373916, 1e-6);
    EXPECT_NEAR(spherical_yn(0, 1.0), -0.5403023058681398, 1e-6);
    EXPECT_NEAR(spherical_in(0, 1.0), 1.1752011936438014, 1e-6);
    EXPECT_NEAR(spherical_kn(0, 1.0), 0.5778636748954609, 1e-3);
}

TEST(SpecialBesselTest, bessel_zeros) {
    EXPECT_NEAR(bessel_zero_jnu(0, 1), 2.4048255576957724, 1e-6);
    EXPECT_NEAR(bessel_zero_jnu(0, 2), 5.520078110286311, 1e-6);
    EXPECT_NEAR(bessel_zero_ynu(0, 1), 0.893576974377206, 1e-3);
}

TEST(SpecialBesselTest, struve_anger_weber) {
    EXPECT_NEAR(struve_h(0, 1.0), 0.5686566270482879, 1e-6);
    EXPECT_NEAR(struve_h(1, 1.0), 0.19845733620194442, 1e-6);
    EXPECT_NEAR(anger_j(1, 1.0), 0.440050585744933, 1e-6);
    EXPECT_NEAR(weber_e(0, 1.0), -0.5686566270482879, 1e-6);
    EXPECT_NEAR(weber_e(1, 1.0), 0.43816243616563694, 1e-3);
}

TEST(SphBesselTest, j0_j1_closed_form) {
    // j_0(x) = sin(x)/x, j_1(x) = sin(x)/x^2 - cos(x)/x; must match to full precision since
    // these ARE the closed forms used internally.
    for (double x : {0.5, 1.0, 2.0, 3.0, M_PI / 2.0}) {
        EXPECT_DOUBLE_EQ(sph_bessel_j(0, x), std::sin(x) / x);
        EXPECT_DOUBLE_EQ(sph_bessel_j(1, x), std::sin(x) / (x * x) - std::cos(x) / x);
    }
}

TEST(SphBesselTest, y0_y1_closed_form) {
    // y_0(x) = -cos(x)/x, y_1(x) = -cos(x)/x^2 - sin(x)/x; exact closed-form match.
    for (double x : {0.5, 1.0, 2.0, 3.0, M_PI / 2.0}) {
        EXPECT_DOUBLE_EQ(sph_bessel_y(0, x), -std::cos(x) / x);
        EXPECT_DOUBLE_EQ(sph_bessel_y(1, x), -std::cos(x) / (x * x) - std::sin(x) / x);
    }
}

TEST(SphBesselTest, j_zero_argument) {
    // j_0(0) = 1 (limit of sin(x)/x); j_n(0) = 0 for n >= 1 (removable-singularity handling).
    EXPECT_EQ(sph_bessel_j(0, 0.0), 1.0);
    for (int n = 1; n <= 6; ++n) {
        EXPECT_EQ(sph_bessel_j(n, 0.0), 0.0) << "n=" << n;
    }
}

TEST(SphBesselTest, y_singularity_at_zero_and_negative_x) {
    // y_n(0) is a genuine pole; this module follows the same convention as bessel_y()/
    // spherical_yn() (its closest siblings) and returns NaN for x <= 0 rather than +/-inf.
    for (int n = 0; n <= 5; ++n) {
        EXPECT_TRUE(std::isnan(sph_bessel_y(n, 0.0))) << "n=" << n;
        EXPECT_TRUE(std::isnan(sph_bessel_y(n, -1.0))) << "n=" << n;
    }
    EXPECT_TRUE(std::isnan(sph_bessel_j(-1, 1.0)));
    EXPECT_TRUE(std::isnan(sph_bessel_y(-1, 1.0)));
}

TEST(SphBesselTest, recurrence_self_consistency) {
    // Independently re-derive j_2..j_4 and y_2..y_4 via the same recurrence relation, written
    // directly here (not by calling sph_bessel_j/sph_bessel_y recursively), and confirm the
    // library function agrees.
    for (double x : {1.0, 2.0, 3.0, 0.5}) {
        const double j0 = std::sin(x) / x;
        const double j1 = std::sin(x) / (x * x) - std::cos(x) / x;
        const double j2 = (3.0 / x) * j1 - j0;
        const double j3 = (5.0 / x) * j2 - j1;
        const double j4 = (7.0 / x) * j3 - j2;
        EXPECT_NEAR(sph_bessel_j(2, x), j2, 1e-12);
        EXPECT_NEAR(sph_bessel_j(3, x), j3, 1e-12);
        EXPECT_NEAR(sph_bessel_j(4, x), j4, 1e-12);

        const double y0 = -std::cos(x) / x;
        const double y1 = -std::cos(x) / (x * x) - std::sin(x) / x;
        const double y2 = (3.0 / x) * y1 - y0;
        const double y3 = (5.0 / x) * y2 - y1;
        const double y4 = (7.0 / x) * y3 - y2;
        EXPECT_NEAR(sph_bessel_y(2, x), y2, 1e-12);
        EXPECT_NEAR(sph_bessel_y(3, x), y3, 1e-12);
        EXPECT_NEAR(sph_bessel_y(4, x), y4, 1e-12);
    }
}

TEST(SphBesselTest, known_cross_check_values) {
    // At x = pi/2: sin(x) = 1, cos(x) = 0, so j_1(pi/2) = 1/(pi/2)^2 = 4/pi^2 and
    // y_1(pi/2) = -1/(pi/2) = -2/pi, by hand from the closed forms.
    const double half_pi = M_PI / 2.0;
    EXPECT_NEAR(sph_bessel_j(1, half_pi), 4.0 / (M_PI * M_PI), 1e-12);
    EXPECT_NEAR(sph_bessel_y(1, half_pi), -2.0 / M_PI, 1e-12);

    // j_2(1.0) cross-checked against j_n(x) = sqrt(pi/(2x)) * J_{n+1/2}(x) with
    // J_{5/2}(1) = sqrt(2/pi) * (3/x^2 - 1) sin(x) - (3/x) cos(x) evaluated at x=1 (DLMF 10.49),
    // and independently against the sibling spherical_jn() implementation.
    EXPECT_NEAR(sph_bessel_j(2, 1.0), 0.062035052011373916, 1e-9);
    EXPECT_NEAR(sph_bessel_j(2, 1.0), spherical_jn(2, 1.0), 1e-9);
}

TEST(SphBesselTest, decay_and_growth_with_order) {
    // For fixed, moderate x, |j_n(x)| generally decreases as n increases (j_n decays with n),
    // while |y_n(x)| generally increases (y_n grows with n) -- the qualitative behavior that
    // makes the shared upward recurrence stable for both directions.
    const double x = 1.0;
    double prevJ = std::abs(sph_bessel_j(0, x));
    double prevY = std::abs(sph_bessel_y(0, x));
    for (int n = 1; n <= 6; ++n) {
        const double curJ = std::abs(sph_bessel_j(n, x));
        const double curY = std::abs(sph_bessel_y(n, x));
        EXPECT_LT(curJ, prevJ) << "n=" << n;
        EXPECT_GT(curY, prevY) << "n=" << n;
        prevJ = curJ;
        prevY = curY;
    }
}

TEST(SphBesselTest, negative_x_parity) {
    // sph_bessel_j is even in x for even n and odd in x for odd n, matching this module's
    // convention for negative arguments of ordinary integer-order Bessel functions
    // (see bessel_j(2, -1.5) == bessel_j(2, 1.5) and bessel_j(3, -1.5) == -bessel_j(3, 1.5)).
    EXPECT_NEAR(sph_bessel_j(0, -1.5), sph_bessel_j(0, 1.5), 1e-12);
    EXPECT_NEAR(sph_bessel_j(2, -1.5), sph_bessel_j(2, 1.5), 1e-12);
    EXPECT_NEAR(sph_bessel_j(1, -1.5), -sph_bessel_j(1, 1.5), 1e-12);
    EXPECT_NEAR(sph_bessel_j(3, -1.5), -sph_bessel_j(3, 1.5), 1e-12);
}

TEST(SpecialBesselTest, kelvin_and_airy_deriv) {
    EXPECT_NEAR(kelvin_ber(0, 1.0), 0.984381781213087, 1e-6);
    EXPECT_NEAR(kelvin_bei(0, 1.0), 0.24956604003665972, 1e-6);
    EXPECT_NEAR(kelvin_ker(0, 1.0), 0.28670620872831604, 1e-6);
    EXPECT_NEAR(kelvin_kei(0, 1.0), -0.49499463651872, 1e-6);
    EXPECT_NEAR(airy_aip(1.0), -0.15914744129679328, 1e-3);
}
