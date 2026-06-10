#include <gtest/gtest.h>

#include "ms/core/operations.hpp"

template<typename T>
class SolveTypedTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SolveTypedTest, RealTypes);

TYPED_TEST(SolveTypedTest, solve_matches_reference) {
    using S = TypeParam;
    auto A = ms::ColMatrix<S>{{S(2), S(1)}, {S(1), S(3)}};
    auto b = ms::ColMatrix<S>{{S(4)}, {S(7)}};
    auto x = ms::solve(A, b);
    ASSERT_TRUE(x.has_value());
    const auto tol = std::is_same_v<S, float> ? 1e-4 : 1e-12;
    EXPECT_NEAR(static_cast<double>((*x)(0, 0)), 1.0, tol);
    EXPECT_NEAR(static_cast<double>((*x)(1, 0)), 2.0, tol);
}
