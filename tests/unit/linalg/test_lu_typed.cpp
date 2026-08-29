#include <complex>
#include <gtest/gtest.h>

#include "ms/core/operations.hpp"

template<typename T>
class LUTypedTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(LUTypedTest, RealTypes);

TYPED_TEST(LUTypedTest, basic_decomposition_PA_equals_LU) {
    using S = TypeParam;
    auto A = ms::ColMatrix<S>{{S(1), S(2), S(3)}, {S(4), S(5), S(6)}, {S(7), S(8), S(10)}};
    auto result = ms::lu(A);
    ASSERT_TRUE(result.has_value());
    auto [L, U, P] = *result;
    auto PA = ms::matmul(P, A);
    auto LU = ms::matmul(L, U);
    ASSERT_TRUE(PA.has_value());
    ASSERT_TRUE(LU.has_value());
    const auto tol = std::is_same_v<S, float> ? S(1e-5) : S(1e-12);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(static_cast<double>((*PA)(i, j)), static_cast<double>((*LU)(i, j)), tol);
        }
    }
}

TYPED_TEST(LUTypedTest, singular_matrix_returns_error) {
    using S = TypeParam;
    auto A = ms::ColMatrix<S>{{S(1), S(2)}, {S(2), S(4)}};
    auto result = ms::lu(A);
    EXPECT_FALSE(result.has_value());
}
