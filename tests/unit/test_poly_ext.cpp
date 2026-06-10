#include <gtest/gtest.h>
#include <cmath>
#include "ms/poly/poly.hpp"

using namespace ms;

TEST(PolyExtTest, eval_constant) {
    // p(x) = 7 -> p(anything) = 7
    std::vector<double> p{7.0};
    EXPECT_DOUBLE_EQ(poly_eval(p, 0.0)[0], 7.0);
    EXPECT_DOUBLE_EQ(poly_eval(p, 100.0)[0], 7.0);
    EXPECT_DOUBLE_EQ(poly_eval(p, -5.0)[0], 7.0);
}

TEST(PolyExtTest, eval_linear) {
    // p(x) = 2 + 3x -> p(4) = 14
    std::vector<double> p{2.0, 3.0};
    EXPECT_NEAR(poly_eval(p, 4.0)[0], 14.0, 1e-12);
    EXPECT_NEAR(poly_eval(p, 0.0)[0], 2.0, 1e-12);
}

TEST(PolyExtTest, eval_quadratic) {
    // p(x) = x^2 - 1 -> p(3) = 8, p(1) = 0, p(-1) = 0
    std::vector<double> p{-1.0, 0.0, 1.0};
    EXPECT_NEAR(poly_eval(p, 3.0)[0], 8.0, 1e-12);
    EXPECT_NEAR(poly_eval(p, 1.0)[0], 0.0, 1e-12);
    EXPECT_NEAR(poly_eval(p, -1.0)[0], 0.0, 1e-12);
}

TEST(PolyExtTest, deriv_quadratic) {
    // p(x) = 1 + 2x + 3x^2 -> p'(x) = 2 + 6x -> p'(2) = 14
    std::vector<double> p{1.0, 2.0, 3.0};
    const auto dp = poly_deriv(p);
    EXPECT_EQ(dp.size(), 2u);
    EXPECT_NEAR(poly_eval(dp, 2.0)[0], 14.0, 1e-12);
}

TEST(PolyExtTest, deriv_constant_is_zero) {
    std::vector<double> p{5.0};
    const auto dp = poly_deriv(p);
    EXPECT_NEAR(poly_eval(dp, 99.0)[0], 0.0, 1e-12);
}

TEST(PolyExtTest, mul_monomials) {
    // x * x = x^2 -> coeffs [0, 0, 1]
    const auto x = std::vector<double>{0.0, 1.0};
    const auto x2 = poly_mul(x, x);
    EXPECT_EQ(x2.size(), 3u);
    EXPECT_NEAR(x2[0], 0.0, 1e-12);
    EXPECT_NEAR(x2[1], 0.0, 1e-12);
    EXPECT_NEAR(x2[2], 1.0, 1e-12);
}

TEST(PolyExtTest, add_and_sub_inverse) {
    const std::vector<double> a{1.0, 2.0, 3.0};
    const std::vector<double> b{4.0, 5.0};
    const auto sum = poly_add(a, b);
    const auto diff = poly_sub(sum, b);
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(diff[i], a[i], 1e-12);
    }
}

TEST(PolyExtTest, eval_negative_coeffs) {
    // p(x) = -x^2 + x -> p(2) = -4 + 2 = -2
    std::vector<double> p{0.0, 1.0, -1.0};
    EXPECT_NEAR(poly_eval(p, 2.0)[0], -2.0, 1e-12);
}
