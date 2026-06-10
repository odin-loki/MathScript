#include <gtest/gtest.h>
#include <cmath>
#include <tuple>

#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

namespace {

double frob(const DMatrix& A) {
    double sum = 0.0;
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            sum += A(i, j) * A(i, j);
    return std::sqrt(sum);
}

DMatrix mat_mul(const DMatrix& A, const DMatrix& B) {
    return matmul<double>(A, B).value();
}

DMatrix eye_n(size_t n) {
    return eye<double>(n);
}

} // namespace

// QR decomposition: A = Q * R
TEST(NumericalQR, square_3x3_reconstruction) {
    DMatrix A(3, 3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 10;  // full rank (det != 0)

    const auto result = qr(A);
    ASSERT_TRUE(result.has_value());
    const auto& [Q, R] = *result;

    EXPECT_EQ(Q.rows(), 3u);
    EXPECT_EQ(Q.cols(), 3u);
    EXPECT_EQ(R.rows(), 3u);
    EXPECT_EQ(R.cols(), 3u);

    // Verify A = Q * R
    const auto QR = mat_mul(Q, R);
    EXPECT_NEAR(frob(QR), frob(A), 1e-9 * frob(A));
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(QR(i,j), A(i,j), 1e-9);
}

TEST(NumericalQR, orthogonality_Q_transpose_Q_is_I) {
    DMatrix A(3, 3);
    A(0,0) = 2; A(0,1) = -1; A(0,2) = 0;
    A(1,0) = -1; A(1,1) = 2; A(1,2) = -1;
    A(2,0) = 0; A(2,1) = -1; A(2,2) = 2;

    const auto result = qr(A);
    ASSERT_TRUE(result.has_value());
    const auto& [Q, R] = *result;

    // Q^T * Q should be identity
    DMatrix Qt(Q.cols(), Q.rows());
    for (size_t i = 0; i < Q.rows(); ++i)
        for (size_t j = 0; j < Q.cols(); ++j)
            Qt(j, i) = Q(i, j);

    const auto QtQ = mat_mul(Qt, Q);
    const auto I = eye_n(Q.cols());
    for (size_t i = 0; i < Q.cols(); ++i)
        for (size_t j = 0; j < Q.cols(); ++j)
            EXPECT_NEAR(QtQ(i, j), I(i, j), 1e-9);
}

TEST(NumericalQR, R_is_upper_triangular) {
    DMatrix A(3, 3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 0; A(1,1) = 4; A(1,2) = 5;
    A(2,0) = 0; A(2,1) = 0; A(2,2) = 6;

    const auto result = qr(A);
    ASSERT_TRUE(result.has_value());
    const auto& [Q, R] = *result;

    // Below-diagonal of R should be ~0
    for (size_t i = 1; i < R.rows(); ++i)
        for (size_t j = 0; j < i; ++j)
            EXPECT_NEAR(R(i, j), 0.0, 1e-9);
}

// logm / expm roundtrip on SPD matrix
TEST(NumericalMatrixFunctions, logm_expm_roundtrip_2x2_spd) {
    DMatrix A(2, 2);
    A(0,0) = 4; A(0,1) = 2;
    A(1,0) = 2; A(1,1) = 3;

    const auto logm_res = logm(A);
    ASSERT_TRUE(logm_res.has_value());

    const auto back = expm(*logm_res);
    ASSERT_TRUE(back.has_value());

    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            EXPECT_NEAR((*back)(i,j), A(i,j), 1e-6);  // logm/expm chain: ~1e-7 tolerance
}

// sqrtm: sqrtm(A)^2 == A for diagonal matrix
TEST(NumericalMatrixFunctions, sqrtm_squared_is_original) {
    DMatrix A = eye<double>(3);
    A(0,0) = 4; A(1,1) = 9; A(2,2) = 16;

    const auto S = sqrtm(A);
    ASSERT_TRUE(S.has_value());

    const auto S2 = mat_mul(*S, *S);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(S2(i,j), A(i,j), 1e-8);
}

// ||A||_2 <= ||A||_F (Frobenius norm upper bounds spectral norm)
TEST(NumericalNorm, spectral_norm_le_frobenius) {
    DMatrix A(3, 3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 9;

    const auto n2 = norm(A, 2);
    ASSERT_TRUE(n2.has_value());

    const double nF = frob(A);
    EXPECT_LE(*n2, nF + 1e-10);  // spectral norm <= Frobenius norm
    EXPECT_GT(*n2, 0.0);
}
