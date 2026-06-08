#include <gtest/gtest.h>

#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"

using namespace ms;

using FMatrix = ColMatrix<float>;

TEST(FloatLinalgTest, matmul_basic) {
    FMatrix A{{1.f, 2.f}, {3.f, 4.f}};
    FMatrix B{{5.f, 6.f}, {7.f, 8.f}};
    const auto C = matmul(A, B).value();
    EXPECT_NEAR(C(0, 0), 19.f, 1e-5f);
    EXPECT_NEAR(C(1, 1), 50.f, 1e-5f);
}

TEST(FloatLinalgTest, solve_via_lu_fallback) {
    FMatrix A{{4.f, 1.f}, {1.f, 3.f}};
    FMatrix b{{1.f}, {2.f}};
    const auto x = solve(A, b).value();
    const FMatrix Ax = A * x;
    EXPECT_NEAR(Ax(0, 0), b(0, 0), 1e-4f);
    EXPECT_NEAR(Ax(1, 0), b(1, 0), 1e-4f);
}

TEST(FloatLinalgTest, lu_and_qr_fallback) {
    FMatrix A{{3.f, 1.f}, {1.f, 2.f}};
    const auto [L, U, P] = lu(A).value();
    const FMatrix prod = P * L * U;
    EXPECT_NEAR(prod(0, 0), A(0, 0), 1e-4f);
    EXPECT_NEAR(prod(1, 1), A(1, 1), 1e-4f);

    const auto [Q, R] = qr(A).value();
    const FMatrix recon = Q * R;
    EXPECT_NEAR(recon(0, 0), A(0, 0), 1e-4f);
    EXPECT_NEAR(recon(1, 1), A(1, 1), 1e-4f);
}

TEST(FloatLinalgTest, float_matmul_matches_reference) {
    FMatrix A{{2.f, 0.f}, {0.f, 3.f}};
    FMatrix B{{4.f, 0.f}, {0.f, 5.f}};
    const auto C = matmul(A, B).value();
    EXPECT_NEAR(C(0, 0), 8.f, 1e-5f);
    EXPECT_NEAR(C(1, 1), 15.f, 1e-5f);
}
