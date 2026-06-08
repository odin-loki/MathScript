#include <gtest/gtest.h>
#include "ms/poly/poly.hpp"
#include "ms/symbolic/symbolic.hpp"
#include "ms/domain/domain.hpp"

using namespace ms;

TEST(PolyTest, eval_and_deriv) {
    std::vector<double> p{1, 2, 3};
    EXPECT_NEAR(poly_eval(p, 2.0)[0], 17.0, 1e-12);
    auto dp = poly_deriv(p);
    EXPECT_NEAR(poly_eval(dp, 2.0)[0], 14.0, 1e-12);
}

TEST(PolyTest, multiply) {
    auto a = std::vector<double>{1, 1};
    auto b = std::vector<double>{1, -1};
    auto c = poly_mul(a, b);
    EXPECT_NEAR(c[0], 1.0, 1e-12);
    EXPECT_NEAR(c[1], 0.0, 1e-12);
    EXPECT_NEAR(c[2], -1.0, 1e-12);
}

TEST(PolyTest, subtract_and_add_mismatch) {
    auto a = std::vector<double>{5.0, 2.0, 1.0};
    auto b = std::vector<double>{1.0, 1.0};
    auto diff = poly_sub(a, b);
    EXPECT_NEAR(diff[0], 4.0, 1e-12);
    EXPECT_NEAR(diff[1], 1.0, 1e-12);
    EXPECT_NEAR(diff[2], 1.0, 1e-12);
    auto sum = poly_add(diff, b);
    EXPECT_NEAR(poly_eval(sum, 0.0)[0], poly_eval(a, 0.0)[0], 1e-12);
}

TEST(PolyTest, edge_empty_and_constant) {
    EXPECT_TRUE(poly_mul({}, {1.0, 2.0}).empty());
    EXPECT_NEAR(poly_eval(poly_deriv({7.0}), 5.0)[0], 0.0, 1e-12);
    EXPECT_NEAR(poly_eval({}, 99.0)[0], 0.0, 1e-12);
}

TEST(SymbolicTest, to_string) {
    auto expr = sym_add(sym_var("x"), sym_const(1.0));
    EXPECT_EQ(sym_to_string(expr), "(x + 1.000000)");
}

TEST(DomainTest, combinatorics) {
    EXPECT_EQ(nchoosek(5, 2), 10);
    EXPECT_EQ(gcd(12, 8), 4);
}
