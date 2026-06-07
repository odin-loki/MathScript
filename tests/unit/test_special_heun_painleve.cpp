#include <gtest/gtest.h>
#include "ms/special/special.hpp"

using namespace ms;

TEST(SpecialHeunTest, general_and_confluent) {
    const double a = 0.5;
    const double q = 0.1;
    const double alpha = 0.2;
    const double beta = 0.3;
    const double gamma = 0.4;
    const double delta = 0.5;
    const double z = 0.2;

    EXPECT_NEAR(heun_g(a, q, alpha, beta, gamma, delta, z), 1.1156353442217855, 5e-3);
    EXPECT_NEAR(heun_c(q, alpha, beta, gamma, delta, z), 0.9472246160936063, 5e-3);
    EXPECT_NEAR(heun_d(q, alpha * beta, gamma, delta, z), 0.9492497291736461, 5e-3);
    EXPECT_NEAR(heun_b(q, alpha, beta, delta, z), 1.0738133528291396, 5e-3);
    EXPECT_NEAR(heun_t(q, alpha, beta, gamma, z), 0.9951038222054902, 5e-3);
    EXPECT_NEAR(heun_g(a, q, alpha, beta, gamma, delta, 1e-6), 1.0, 1e-12);
}

TEST(SpecialPainleveTest, first_and_second) {
    EXPECT_NEAR(painleve1(0.5, 0.0, 0.0), -0.020821712245422376, 1e-6);
    EXPECT_NEAR(painleve1(1.0, 0.0, 0.0), -0.16372821373317917, 1e-6);
    EXPECT_NEAR(painleve2(0.5, 0.0, -0.5, 0.0), -0.2530083397481052, 1e-6);
    EXPECT_NEAR(painleve1(0.0, 0.0, 0.0), 0.0, 1e-15);
}

TEST(SpecialPainleveTest, third_through_sixth) {
    EXPECT_NEAR(painleve3(0.5, 0.5, -0.1, 0.5, 0.3), 1.398748842793728, 1e-3);
    EXPECT_NEAR(painleve4(0.5, 0.8, -0.05, 0.2, 0.4), 0.786121344510419, 1e-3);
    EXPECT_NEAR(painleve5(0.5, 0.5, -0.05, 0.01, 0.02, 0.03, 0.04), 0.5558194327648597, 1e-3);
    EXPECT_NEAR(painleve6(2.5, 0.5, -0.05, 0.1, 0.2, 0.3, 0.4), 0.5003268969869713, 1e-3);
}
