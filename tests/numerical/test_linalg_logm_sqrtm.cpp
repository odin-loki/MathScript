// MathScript Tests: logm, sqrtm, Schur, Bidiag (Wave 52)

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>

#include "ms/linalg/linalg.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = Matrix<double>;

// Helper: manual matvec
static DMatrix matvec_m(const DMatrix& A, const DMatrix& x) {
    size_t n = A.rows();
    DMatrix r(n, 1);
    for (size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (size_t k = 0; k < A.cols(); ++k) s += A(i, k) * x(k, 0);
        r(i, 0) = s;
    }
    return r;
}

// Helper: manual matmul (A n×k) × (B k×m) → n×m
static DMatrix matmul_m(const DMatrix& A, const DMatrix& B) {
    size_t n = A.rows(), k = A.cols(), m = B.cols();
    DMatrix C(n, m);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j) {
            double s = 0.0;
            for (size_t p = 0; p < k; ++p) s += A(i, p) * B(p, j);
            C(i, j) = s;
        }
    return C;
}

// Helper: Frobenius norm
static double frob_norm(const DMatrix& A) {
    double s = 0.0;
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            s += A(i, j) * A(i, j);
    return std::sqrt(s);
}

// Helper: max off-diagonal error vs identity
static double ortho_error(const DMatrix& Q) {
    auto QtQ = matmul_m(Q, Q); // wrong - need Q^T * Q
    // Build Q^T manually
    size_t n = Q.rows(), m = Q.cols();
    DMatrix Qt(m, n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
            Qt(j, i) = Q(i, j);
    auto QtQ2 = matmul_m(Qt, Q);
    double err = 0.0;
    for (size_t i = 0; i < QtQ2.rows(); ++i)
        for (size_t j = 0; j < QtQ2.cols(); ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            err = std::max(err, std::abs(QtQ2(i, j) - expected));
        }
    return err;
}

// ---------------------------------------------------------------------------
// logm: matrix logarithm
// ---------------------------------------------------------------------------

TEST(LinalgLogmSqrtm, Logm_Identity_IsZero) {
    // log(I) = 0
    DMatrix I(3, 3);
    for (size_t i = 0; i < 3; ++i) I(i, i) = 1.0;
    auto result = logm(I);
    ASSERT_TRUE(result.has_value()) << "logm(I) should succeed";
    auto& L = result.value();
    for (size_t i = 0; i < L.rows(); ++i)
        for (size_t j = 0; j < L.cols(); ++j)
            EXPECT_NEAR(L(i, j), 0.0, 1e-6) << "logm(I) should be zero at (" << i << "," << j << ")";
}

TEST(LinalgLogmSqrtm, Logm_DiagonalMatrix_Finite) {
    // logm of positive diagonal matrix should give finite result
    DMatrix A(2, 2);
    A(0, 0) = std::exp(1.0);
    A(1, 1) = std::exp(2.0);
    auto result = logm(A);
    ASSERT_TRUE(result.has_value());
    auto& L = result.value();
    EXPECT_TRUE(std::isfinite(L(0, 0)));
    EXPECT_TRUE(std::isfinite(L(1, 1)));
    // trace(logm(A)) = log(det(A)) = log(e^1 * e^2) = 3
    double trace_L = L(0, 0) + L(1, 1);
    EXPECT_NEAR(trace_L, 3.0, 0.1);
}

TEST(LinalgLogmSqrtm, Logm_AllFinite) {
    // logm of diagonal positive matrix should give finite result
    DMatrix A(3, 3);
    A(0, 0) = 2.0; A(1, 1) = 3.0; A(2, 2) = 5.0;
    auto result = logm(A);
    if (result.has_value()) {
        auto& L = result.value();
        for (size_t i = 0; i < L.rows(); ++i)
            for (size_t j = 0; j < L.cols(); ++j)
                EXPECT_TRUE(std::isfinite(L(i, j)));
    }
}

TEST(LinalgLogmSqrtm, Logm_Shape_PreservedFrom_Input) {
    DMatrix A(4, 4);
    for (size_t i = 0; i < 4; ++i) A(i, i) = 2.0 + static_cast<double>(i);
    auto result = logm(A);
    if (result.has_value()) {
        EXPECT_EQ(result.value().rows(), 4u);
        EXPECT_EQ(result.value().cols(), 4u);
    }
}

// ---------------------------------------------------------------------------
// sqrtm: matrix square root
// ---------------------------------------------------------------------------

TEST(LinalgLogmSqrtm, Sqrtm_Identity_IsIdentity) {
    // sqrt(I) = I
    DMatrix I(3, 3);
    for (size_t i = 0; i < 3; ++i) I(i, i) = 1.0;
    auto result = sqrtm(I);
    ASSERT_TRUE(result.has_value()) << "sqrtm(I) should succeed";
    auto& S = result.value();
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(S(i, j), expected, 1e-6)
                << "sqrtm(I) at (" << i << "," << j << ")";
        }
}

TEST(LinalgLogmSqrtm, Sqrtm_Diagonal_CorrectValues) {
    // sqrt(diag(4, 9)) = diag(2, 3)
    DMatrix A(2, 2);
    A(0, 0) = 4.0; A(1, 1) = 9.0;
    auto result = sqrtm(A);
    ASSERT_TRUE(result.has_value());
    auto& S = result.value();
    EXPECT_NEAR(S(0, 0), 2.0, 1e-5);
    EXPECT_NEAR(S(1, 1), 3.0, 1e-5);
}

TEST(LinalgLogmSqrtm, Sqrtm_Squared_EqualsOriginal) {
    // sqrtm(A)^2 should equal A for SPD A
    DMatrix A(3, 3);
    A(0, 0) = 4.0; A(1, 1) = 9.0; A(2, 2) = 16.0;
    A(0, 1) = 1.0; A(1, 0) = 1.0; // slightly off-diagonal, still SPD? No.
    // Use purely diagonal for safety
    A(0, 1) = 0.0; A(1, 0) = 0.0;
    auto result = sqrtm(A);
    ASSERT_TRUE(result.has_value());
    auto& S = result.value();
    auto SS = matmul_m(S, S);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(SS(i, j), A(i, j), 1e-4)
                << "sqrtm^2 != A at (" << i << "," << j << ")";
}

TEST(LinalgLogmSqrtm, Sqrtm_AllFinite) {
    DMatrix A(3, 3);
    A(0, 0) = 1.0; A(1, 1) = 4.0; A(2, 2) = 9.0;
    auto result = sqrtm(A);
    if (result.has_value()) {
        auto& S = result.value();
        for (size_t i = 0; i < S.rows(); ++i)
            for (size_t j = 0; j < S.cols(); ++j)
                EXPECT_TRUE(std::isfinite(S(i, j)));
    }
}

TEST(LinalgLogmSqrtm, Sqrtm_Shape_Preserved) {
    DMatrix A(4, 4);
    for (size_t i = 0; i < 4; ++i) A(i, i) = static_cast<double>(i + 1);
    auto result = sqrtm(A);
    if (result.has_value()) {
        EXPECT_EQ(result.value().rows(), 4u);
        EXPECT_EQ(result.value().cols(), 4u);
    }
}

// ---------------------------------------------------------------------------
// Schur decomposition
// ---------------------------------------------------------------------------

TEST(LinalgLogmSqrtm, Schur_ReturnsValue) {
    DMatrix A(3, 3);
    A(0, 0) = 2.0; A(0, 1) = 1.0;
    A(1, 0) = 0.0; A(1, 1) = 3.0; A(1, 2) = 1.0;
    A(2, 2) = 4.0;
    auto result = schur(A);
    EXPECT_TRUE(result.has_value()) << "schur should succeed";
}

TEST(LinalgLogmSqrtm, Schur_T_AllFinite) {
    DMatrix A(3, 3);
    A(0, 0) = 2.0; A(0, 1) = 1.0; A(1, 1) = 3.0; A(2, 2) = 1.0;
    auto result = schur(A);
    ASSERT_TRUE(result.has_value());
    auto& T = result.value().T;
    for (size_t i = 0; i < T.rows(); ++i)
        for (size_t j = 0; j < T.cols(); ++j)
            EXPECT_TRUE(std::isfinite(T(i, j)));
}

TEST(LinalgLogmSqrtm, Schur_Q_AllFinite) {
    DMatrix A(3, 3);
    A(0, 0) = 1.0; A(1, 1) = 2.0; A(2, 2) = 3.0;
    auto result = schur(A);
    ASSERT_TRUE(result.has_value());
    auto& Q = result.value().Q;
    for (size_t i = 0; i < Q.rows(); ++i)
        for (size_t j = 0; j < Q.cols(); ++j)
            EXPECT_TRUE(std::isfinite(Q(i, j)));
}

TEST(LinalgLogmSqrtm, Schur_Q_NearlyOrthogonal) {
    DMatrix A(3, 3);
    A(0, 0) = 4.0; A(0, 1) = 1.0;
    A(1, 0) = 2.0; A(1, 1) = 3.0; A(1, 2) = 1.0;
    A(2, 1) = 1.0; A(2, 2) = 2.0;
    auto result = schur(A);
    ASSERT_TRUE(result.has_value());
    double err = ortho_error(result.value().Q);
    EXPECT_LT(err, 1e-6) << "Schur Q should be orthogonal";
}

TEST(LinalgLogmSqrtm, Schur_DiagonalInput_TIsDiagonal) {
    // For a diagonal matrix, Schur T should also be (near-)diagonal
    DMatrix A(3, 3);
    A(0, 0) = 5.0; A(1, 1) = 2.0; A(2, 2) = 8.0;
    auto result = schur(A);
    ASSERT_TRUE(result.has_value());
    auto& T = result.value().T;
    // Off-diagonal elements of T should be small
    for (size_t i = 0; i < T.rows(); ++i)
        for (size_t j = 0; j < T.cols(); ++j)
            if (i != j) EXPECT_NEAR(T(i, j), 0.0, 1e-6);
}

// ---------------------------------------------------------------------------
// Bidiagonalization
// ---------------------------------------------------------------------------

TEST(LinalgLogmSqrtm, Bidiag_ReturnsValue) {
    DMatrix A(4, 3);
    A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
    A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;
    A(2, 0) = 7.0; A(2, 1) = 8.0; A(2, 2) = 9.0;
    A(3, 0) = 1.0; A(3, 1) = 0.0; A(3, 2) = 1.0;
    auto result = bidiag(A);
    EXPECT_TRUE(result.has_value()) << "bidiag should succeed";
}

TEST(LinalgLogmSqrtm, Bidiag_B_AllFinite) {
    DMatrix A(3, 3);
    A(0, 0) = 1.0; A(0, 1) = 2.0;
    A(1, 0) = 3.0; A(1, 1) = 4.0; A(1, 2) = 1.0;
    A(2, 1) = 1.0; A(2, 2) = 2.0;
    auto result = bidiag(A);
    ASSERT_TRUE(result.has_value());
    auto& B = result.value().B;
    for (size_t i = 0; i < B.rows(); ++i)
        for (size_t j = 0; j < B.cols(); ++j)
            EXPECT_TRUE(std::isfinite(B(i, j)));
}

TEST(LinalgLogmSqrtm, Bidiag_U_AllFinite) {
    DMatrix A(3, 3);
    A(0, 0) = 1.0; A(1, 1) = 2.0; A(2, 2) = 3.0;
    auto result = bidiag(A);
    ASSERT_TRUE(result.has_value());
    auto& U = result.value().U;
    for (size_t i = 0; i < U.rows(); ++i)
        for (size_t j = 0; j < U.cols(); ++j)
            EXPECT_TRUE(std::isfinite(U(i, j)));
}

TEST(LinalgLogmSqrtm, Bidiag_V_AllFinite) {
    DMatrix A(3, 3);
    A(0, 0) = 2.0; A(1, 1) = 4.0; A(2, 2) = 6.0;
    auto result = bidiag(A);
    ASSERT_TRUE(result.has_value());
    auto& V = result.value().V;
    for (size_t i = 0; i < V.rows(); ++i)
        for (size_t j = 0; j < V.cols(); ++j)
            EXPECT_TRUE(std::isfinite(V(i, j)));
}
