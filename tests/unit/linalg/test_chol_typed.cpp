#include <gtest/gtest.h>

#include "ms/core/operations.hpp"

template<typename T>
class CholTypedTest : public ::testing::Test {};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(CholTypedTest, RealTypes);

TYPED_TEST(CholTypedTest, spd_roundtrip) {
    using S = TypeParam;
    auto A = ms::ColMatrix<S>{{S(4), S(1)}, {S(1), S(3)}};
    const auto L = ms::chol(A);
    ASSERT_TRUE(L.has_value());
    const auto recon = (*L) * ms::transpose(*L);
    const auto tol = std::is_same_v<S, float> ? S(1e-5) : S(1e-9);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(static_cast<double>(recon(i, j)), static_cast<double>(A(i, j)), tol);
        }
    }
}

TYPED_TEST(CholTypedTest, non_spd_returns_error) {
    using S = TypeParam;
    auto A = ms::ColMatrix<S>{{S(1), S(2)}, {S(3), S(4)}};
    EXPECT_FALSE(ms::chol(A).has_value());
}
