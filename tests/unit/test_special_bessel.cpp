#include <gtest/gtest.h>
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

TEST(SpecialBesselTest, kelvin_and_airy_deriv) {
    EXPECT_NEAR(kelvin_ber(0, 1.0), 0.984381781213087, 1e-6);
    EXPECT_NEAR(kelvin_bei(0, 1.0), 0.24956604003665972, 1e-6);
    EXPECT_NEAR(kelvin_ker(0, 1.0), 0.28670620872831604, 1e-6);
    EXPECT_NEAR(kelvin_kei(0, 1.0), -0.49499463651872, 1e-6);
    EXPECT_NEAR(airy_aip(1.0), -0.15914744129679328, 1e-3);
}
