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

TEST(SpecialHyperTest, meijer_and_fox_scaffold) {
    EXPECT_NEAR(meijer_g(1.0, 2.0, 0.5), 0.39346934028736663, 1e-3);
    EXPECT_NEAR(fox_h(1.0, 2.0, 0.5), meijer_g(1.0, 2.0, 0.5), 1e-12);
}
