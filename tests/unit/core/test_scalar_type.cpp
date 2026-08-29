#include <gtest/gtest.h>
#include "ms/core/scalar.hpp"

using namespace ms;

TEST(ScalarTypeTest, default_construction) {
    const Scalar s;
    EXPECT_DOUBLE_EQ(s.value(), 0.0);
    EXPECT_TRUE(s.unit().empty());
}

TEST(ScalarTypeTest, value_and_unit) {
    const Scalar s(3.14, "m");
    EXPECT_DOUBLE_EQ(s.value(), 3.14);
    EXPECT_EQ(s.unit(), "m");
}

TEST(ScalarTypeTest, addition) {
    const Scalar a(1.5, "kg"), b(2.5, "kg");
    const auto c = a + b;
    EXPECT_DOUBLE_EQ(c.value(), 4.0);
    EXPECT_EQ(c.unit(), "kg");
}

TEST(ScalarTypeTest, subtraction) {
    const Scalar a(5.0, "s"), b(2.0, "s");
    const auto c = a - b;
    EXPECT_DOUBLE_EQ(c.value(), 3.0);
}

TEST(ScalarTypeTest, multiplication) {
    const Scalar a(3.0, "m"), b(2.0, "s");
    const auto c = a * b;
    EXPECT_DOUBLE_EQ(c.value(), 6.0);
}

TEST(ScalarTypeTest, division) {
    const Scalar a(9.0, "m"), b(3.0, "s");
    const auto c = a / b;
    EXPECT_DOUBLE_EQ(c.value(), 3.0);
}

TEST(ScalarTypeTest, prefix_full_unit) {
    const Scalar s(1.0, "m", "k");
    EXPECT_EQ(s.full_unit(), "km");
}

TEST(ScalarTypeTest, zero_value) {
    const Scalar s(0.0, "Pa");
    const Scalar other(5.0, "Pa");
    const auto result = s + other;
    EXPECT_DOUBLE_EQ(result.value(), 5.0);
}
