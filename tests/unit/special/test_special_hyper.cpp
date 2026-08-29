#include <gtest/gtest.h>
#include <cmath>
#include "ms/special/special.hpp"

using namespace ms;

TEST(SpecialHyperTest, confluent_hypergeometric) {
    EXPECT_NEAR(hypergeo_0f1(2.0, 1.0), 1.590636854637329, 1e-3);
    EXPECT_NEAR(kummer_m(1.0, 2.0, 0.5), 1.2974425414002564, 1e-3);
    EXPECT_NEAR(hypergeo_1f1(1.0, 0.0), 1.0, 1e-12);
}

TEST(SpecialHyperTest, gauss_hypergeometric) {
    EXPECT_NEAR(hypergeo_2f1(1.0, 1.0, 2.0, 0.5), -std::log(0.5) / 0.5, 1e-6);
    EXPECT_NEAR(hypergeo_2f1(2.0, 1.0, 1.0, 0.3), 2.0408163265306123, 1e-6);
}

TEST(SpecialHyperTest, whittaker_functions) {
    EXPECT_NEAR(whittaker_m(0.0, 0.5, 1.0), 1.0421906109874948, 1e-3);
    EXPECT_NEAR(whittaker_w(0.0, 0.5, 1.0), 0.6065306597126334, 1e-3);
}

TEST(SpecialHyperTest, tricomi_u) {
    EXPECT_NEAR(tricomi_u(1.0, 2.0, 0.5), 2.0, 1e-3);
    EXPECT_NEAR(tricomi_u(0.5, 1.5, 1.0), 1.0, 1e-3);
    EXPECT_NEAR(tricomi_u(2.0, 3.0, 0.2), 25.0, 1e-3);
}

namespace {

double kummer_u_connection_ref(double a, double b, double z) {
    const double log_c1 = log_gamma(1.0 - b) - log_gamma(a - b + 1.0);
    const double log_c2 = log_gamma(b - 1.0) - log_gamma(a);
    return std::exp(log_c1) * kummer_m(a, b, z) - std::exp(log_c2) * std::pow(z, 1.0 - b) *
           kummer_m(a - b + 1.0, 2.0 - b, z);
}

} // namespace

TEST(SpecialHyperTest, kummer_u_special_case_a_plus_one) {
    EXPECT_NEAR(kummer_u(0.5, 1.5, 2.0), std::pow(2.0, -0.5), 1e-12);
    EXPECT_NEAR(kummer_u(1.0, 2.0, 0.5), 2.0, 1e-12);
    EXPECT_NEAR(kummer_u(2.0, 3.0, 0.2), 25.0, 1e-12);
}

TEST(SpecialHyperTest, kummer_u_linear_independence_from_m) {
    const double a = 0.5;
    const double b = 1.5;
    const double ratio_z1 = kummer_u(a, b, 0.5) / kummer_m(a, b, 0.5);
    const double ratio_z2 = kummer_u(a, b, 2.0) / kummer_m(a, b, 2.0);
    EXPECT_NE(ratio_z1, ratio_z2);
    EXPECT_TRUE(std::isfinite(ratio_z1));
    EXPECT_TRUE(std::isfinite(ratio_z2));
}

TEST(SpecialHyperTest, kummer_u_monotone_decreasing) {
    const double a = 0.5;
    const double b = 1.5;
    const double u1 = kummer_u(a, b, 1.0);
    const double u2 = kummer_u(a, b, 2.0);
    const double u3 = kummer_u(a, b, 5.0);
    EXPECT_GT(u1, u2);
    EXPECT_GT(u2, u3);
    EXPECT_GT(u1, 0.0);
    EXPECT_GT(u2, 0.0);
    EXPECT_GT(u3, 0.0);
}

TEST(SpecialHyperTest, kummer_u_connection_formula) {
    EXPECT_NEAR(kummer_u(0.2, 1.5, 0.3), kummer_u_connection_ref(0.2, 1.5, 0.3), 1e-3);
    EXPECT_NEAR(kummer_u(0.5, 1.8, 0.4), kummer_u_connection_ref(0.5, 1.8, 0.4), 1e-3);
    EXPECT_NEAR(kummer_u(1.0, 1.5, 0.5), kummer_u_connection_ref(1.0, 1.5, 0.5), 1e-3);
}

TEST(SpecialHyperTest, kummer_u_large_z_asymptotic) {
    const double a = 0.5;
    const double b = 1.5;
    const double z = 25.0;
    EXPECT_NEAR(kummer_u(a, b, z), std::pow(z, -a), 1e-3);
}

TEST(SpecialHyperTest, kummer_u_degenerate_inputs) {
    EXPECT_TRUE(std::isnan(kummer_u(0.5, 1.5, 0.0)));
    EXPECT_TRUE(std::isnan(kummer_u(0.5, 1.5, -0.1)));
    EXPECT_TRUE(std::isnan(kummer_u(0.5, 0.0, 0.5)));
    EXPECT_TRUE(std::isnan(kummer_u(0.5, -1.0, 0.5)));
}

TEST(SpecialHyperTest, meijer_and_fox_scaffold) {
    EXPECT_NEAR(meijer_g(1.0, 2.0, 0.5), 0.39346934028736663, 1e-3);
    EXPECT_NEAR(fox_h(1.0, 2.0, 0.5), meijer_g(1.0, 2.0, 0.5), 1e-12);
}
