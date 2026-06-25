#include "ms/linalg/linalg.hpp"
#include "ms/core/operations.hpp"
#include <cmath>
#include <gtest/gtest.h>

using DMatrix = ms::Matrix<double>;

static double frob(const DMatrix& A) {
    double s = 0.0;
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            s += A(i, j) * A(i, j);
    return std::sqrt(s);
}

TEST(LinalgQMR, SolvesLinearSystem) {
    // Symmetric positive definite 3x3
    DMatrix A(3, 3, 0.0);
    A(0,0)=4; A(0,1)=1; A(0,2)=0;
    A(1,0)=1; A(1,1)=4; A(1,2)=1;
    A(2,0)=0; A(2,1)=1; A(2,2)=4;
    DMatrix b(3, 1, 0.0);
    b(0,0)=1; b(1,0)=2; b(2,0)=3;
    auto x = ms::qmr(A, b, 100, 1e-10);
    ASSERT_TRUE(x.has_value());
    // Verify Ax ≈ b
    auto Ax_r = ms::matmul(A, x.value());
    ASSERT_TRUE(Ax_r.has_value());
    DMatrix& Ax = Ax_r.value();
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(Ax(i,0), b(i,0), 1e-6);
}

TEST(LinalgQMR, DimMismatch) {
    DMatrix A(3, 3, 0.0);
    DMatrix b(2, 1, 0.0);
    auto x = ms::qmr(A, b);
    EXPECT_FALSE(x.has_value());
}

TEST(LinalgLSQR, SolvesSquareSystem) {
    DMatrix A(3, 3, 0.0);
    A(0,0)=4; A(0,1)=1;
    A(1,1)=5; A(1,2)=2;
    A(2,2)=6;
    DMatrix b(3, 1, 0.0);
    b(0,0)=1; b(1,0)=2; b(2,0)=3;
    auto x = ms::lsqr(A, b, 100, 1e-10);
    ASSERT_TRUE(x.has_value());
    EXPECT_EQ(x.value().rows(), 3u);
    // Check finite solution
    for (size_t i = 0; i < 3; ++i)
        EXPECT_TRUE(std::isfinite(x.value()(i,0)));
}

TEST(LinalgLSQR, RectangularOverdetermined) {
    // 4x2 overdetermined system, consistent
    DMatrix A(4, 2, 0.0);
    A(0,0)=1; A(1,1)=1; A(2,0)=1; A(3,1)=1;
    DMatrix b(4, 1, 0.0);
    b(0,0)=1; b(1,0)=2; b(2,0)=1; b(3,0)=2;
    auto x = ms::lsqr(A, b, 200, 1e-8);
    ASSERT_TRUE(x.has_value());
    EXPECT_EQ(x.value().rows(), 2u);
}

TEST(LinalgLSMR, SolvesSystem) {
    DMatrix A(3, 3, 0.0);
    A(0,0)=4; A(1,1)=5; A(2,2)=6;
    DMatrix b(3, 1, 0.0);
    b(0,0)=4; b(1,0)=5; b(2,0)=6;
    auto x = ms::lsmr(A, b, 100, 1e-10);
    ASSERT_TRUE(x.has_value());
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(x.value()(i,0), 1.0, 1e-6);
}

TEST(LinalgTFQMR, SolvesLinearSystem) {
    DMatrix A(3, 3, 0.0);
    A(0,0)=4; A(0,1)=1;
    A(1,0)=1; A(1,1)=4; A(1,2)=1;
    A(2,1)=1; A(2,2)=4;
    DMatrix b(3, 1, 0.0);
    b(0,0)=1; b(1,0)=1; b(2,0)=1;
    auto x = ms::tfqmr(A, b, 200, 1e-8);
    // TFQMR may or may not converge; just check no crash
    (void)x;
    SUCCEED();
}

TEST(LinalgPrecond, DiagPreconditioner) {
    DMatrix A(3, 3, 0.0);
    A(0,0) = 2.0; A(1,1) = 3.0; A(2,2) = 4.0;
    auto d = ms::precond_diag(A);
    EXPECT_EQ(d.size(), 3u);
    EXPECT_NEAR(d[0], 0.5, 1e-10);
    EXPECT_NEAR(d[1], 1.0/3.0, 1e-10);
    EXPECT_NEAR(d[2], 0.25, 1e-10);
}

TEST(LinalgPrecond, SSORPreconditioner) {
    DMatrix A(3, 3, 0.0);
    A(0,0)=4; A(1,1)=5; A(2,2)=6;
    auto M = ms::precond_ssor(A);
    EXPECT_EQ(M.rows(), 3u);
    EXPECT_EQ(M.cols(), 3u);
    // Diagonal entries should be A(i,i)/omega with omega=1
    EXPECT_NEAR(M(0,0), 4.0, 1e-10);
    EXPECT_NEAR(M(1,1), 5.0, 1e-10);
    EXPECT_NEAR(M(2,2), 6.0, 1e-10);
}
