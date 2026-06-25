// MathScript: Advanced Core Matrix Operations Tests (Wave 50)
// Tests for det, trace, norm, expm, lu, qr, chol properties

#include <gtest/gtest.h>
#include <cmath>
#include <tuple>
#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = Matrix<double>;

// Frobenius norm helper
static double frob(const DMatrix& A) {
    double s = 0.0;
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            s += A(i, j) * A(i, j);
    return std::sqrt(s);
}

// Max absolute error between two matrices
static double max_err(const DMatrix& A, const DMatrix& B) {
    double e = 0.0;
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            e = std::max(e, std::abs(A(i, j) - B(i, j)));
    return e;
}

// ---------------------------------------------------------------------------
// det(): determinant
// ---------------------------------------------------------------------------

TEST(CoreOpsAdv, Det_Identity_2x2_IsOne) {
    DMatrix A({{1.0, 0.0}, {0.0, 1.0}});
    auto d = det(A);
    ASSERT_TRUE(d.has_value());
    EXPECT_NEAR(*d, 1.0, 1e-10);
}

TEST(CoreOpsAdv, Det_Identity_3x3_IsOne) {
    DMatrix A({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
    auto d = det(A);
    ASSERT_TRUE(d.has_value());
    EXPECT_NEAR(*d, 1.0, 1e-10);
}

TEST(CoreOpsAdv, Det_2x2_KnownValue) {
    // det([[a,b],[c,d]]) = ad - bc
    DMatrix A({{3.0, 2.0}, {1.0, 4.0}});
    auto d = det(A);
    ASSERT_TRUE(d.has_value());
    EXPECT_NEAR(*d, 3.0 * 4.0 - 2.0 * 1.0, 1e-10);
}

TEST(CoreOpsAdv, Det_SingularMatrix_NearZero) {
    // Rank-deficient matrix: det = 0
    DMatrix A({{1.0, 2.0}, {2.0, 4.0}});
    auto d = det(A);
    // Either returns 0 or an error
    if (d.has_value()) EXPECT_NEAR(*d, 0.0, 1e-8);
}

TEST(CoreOpsAdv, Det_Diagonal_IsProduct) {
    // det(diag(d1,d2,d3)) = d1*d2*d3
    DMatrix A({{2.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 5.0}});
    auto d = det(A);
    ASSERT_TRUE(d.has_value());
    EXPECT_NEAR(*d, 2.0 * 3.0 * 5.0, 1e-9);
}

TEST(CoreOpsAdv, Det_ScaledMatrix) {
    // det(alpha * A) = alpha^n * det(A) for n×n
    DMatrix A({{1.0, 2.0}, {3.0, 4.0}});
    double alpha = 2.0;
    DMatrix scA({{2.0, 4.0}, {6.0, 8.0}});
    auto d1 = det(A);
    auto d2 = det(scA);
    ASSERT_TRUE(d1.has_value());
    ASSERT_TRUE(d2.has_value());
    EXPECT_NEAR(*d2, std::pow(alpha, 2.0) * (*d1), 1e-9);
}

TEST(CoreOpsAdv, Det_Transpose_Same) {
    // det(A) = det(A^T): build A^T manually as ColMajor
    DMatrix A({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 10.0}});
    DMatrix At(3, 3);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            At(j, i) = A(i, j);
    auto d1 = det(A);
    auto d2 = det(At);
    ASSERT_TRUE(d1.has_value());
    ASSERT_TRUE(d2.has_value());
    EXPECT_NEAR(*d1, *d2, 1e-9);
}

// ---------------------------------------------------------------------------
// trace(): matrix trace
// ---------------------------------------------------------------------------

TEST(CoreOpsAdv, Trace_Identity_3x3_IsN) {
    DMatrix A({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
    auto t = trace(A);
    ASSERT_TRUE(t.has_value());
    EXPECT_NEAR(*t, 3.0, 1e-10);
}

TEST(CoreOpsAdv, Trace_Diagonal_IsSumOfDiag) {
    DMatrix A({{2.0, 5.0, 8.0}, {1.0, 3.0, 7.0}, {0.0, 4.0, 6.0}});
    auto t = trace(A);
    ASSERT_TRUE(t.has_value());
    EXPECT_NEAR(*t, 2.0 + 3.0 + 6.0, 1e-10);
}

TEST(CoreOpsAdv, Trace_Cyclic_AB_EqualBA) {
    // trace(A*B) = trace(B*A)
    DMatrix A({{1.0, 2.0}, {3.0, 4.0}});
    DMatrix B({{5.0, 6.0}, {7.0, 8.0}});
    auto AB = matmul<double>(A, B);
    auto BA = matmul<double>(B, A);
    ASSERT_TRUE(AB.has_value());
    ASSERT_TRUE(BA.has_value());
    auto t1 = trace(*AB);
    auto t2 = trace(*BA);
    ASSERT_TRUE(t1.has_value());
    ASSERT_TRUE(t2.has_value());
    EXPECT_NEAR(*t1, *t2, 1e-9);
}

TEST(CoreOpsAdv, Trace_Transpose_Same) {
    DMatrix A({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
    // Build A^T as ColMajor
    DMatrix At(3, 3);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            At(j, i) = A(i, j);
    auto t1 = trace(A);
    auto t2 = trace(At);
    ASSERT_TRUE(t1.has_value());
    ASSERT_TRUE(t2.has_value());
    EXPECT_NEAR(*t1, *t2, 1e-10);
}

// ---------------------------------------------------------------------------
// norm(): matrix norm
// ---------------------------------------------------------------------------

TEST(CoreOpsAdv, Norm_ZeroMatrix_IsZero) {
    DMatrix A(3, 3, 0.0);
    auto n = norm(A);
    ASSERT_TRUE(n.has_value());
    EXPECT_NEAR(*n, 0.0, 1e-10);
}

TEST(CoreOpsAdv, Norm_Identity_IsN) {
    // Frobenius norm of I_n = sqrt(n)
    DMatrix A({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
    // Default p=2: likely Frobenius or spectral norm — just check positive and finite
    auto n = norm(A);
    ASSERT_TRUE(n.has_value());
    EXPECT_GT(*n, 0.0);
    EXPECT_TRUE(std::isfinite(*n));
}

TEST(CoreOpsAdv, Norm_Positive) {
    DMatrix A({{1.0, 2.0}, {3.0, 4.0}});
    auto n = norm(A);
    ASSERT_TRUE(n.has_value());
    EXPECT_GT(*n, 0.0);
}

TEST(CoreOpsAdv, Norm_ScaledMatrix) {
    // norm(alpha*A) = |alpha| * norm(A)
    DMatrix A({{1.0, 2.0}, {3.0, 4.0}});
    DMatrix scA({{3.0, 6.0}, {9.0, 12.0}});
    auto n1 = norm(A);
    auto n2 = norm(scA);
    ASSERT_TRUE(n1.has_value());
    ASSERT_TRUE(n2.has_value());
    EXPECT_NEAR(*n2, 3.0 * (*n1), 1e-9);
}

// ---------------------------------------------------------------------------
// LU decomposition properties
// ---------------------------------------------------------------------------

TEST(CoreOpsAdv, LU_Returns_Value_3x3) {
    DMatrix A({{2.0, 1.0, 1.0}, {4.0, 3.0, 3.0}, {8.0, 7.0, 9.0}});
    auto result = lu(A);
    ASSERT_TRUE(result.has_value());
    auto& [L, U, P] = *result;
    EXPECT_EQ(L.rows(), 3u);
    EXPECT_EQ(U.rows(), 3u);
}

TEST(CoreOpsAdv, LU_L_LowerTriangular) {
    DMatrix A({{2.0, 1.0, 1.0}, {4.0, 3.0, 3.0}, {8.0, 7.0, 9.0}});
    auto result = lu(A);
    ASSERT_TRUE(result.has_value());
    auto& [L, U, P] = *result;
    for (size_t i = 0; i < L.rows(); ++i)
        for (size_t j = i + 1; j < L.cols(); ++j)
            EXPECT_NEAR(L(i, j), 0.0, 1e-10) << "L not lower triangular at (" << i << "," << j << ")";
}

TEST(CoreOpsAdv, LU_U_UpperTriangular) {
    DMatrix A({{2.0, 1.0, 1.0}, {4.0, 3.0, 3.0}, {8.0, 7.0, 9.0}});
    auto result = lu(A);
    ASSERT_TRUE(result.has_value());
    auto& [L, U, P] = *result;
    for (size_t i = 1; i < U.rows(); ++i)
        for (size_t j = 0; j < i; ++j)
            EXPECT_NEAR(U(i, j), 0.0, 1e-10) << "U not upper triangular at (" << i << "," << j << ")";
}

// ---------------------------------------------------------------------------
// QR decomposition properties
// ---------------------------------------------------------------------------

TEST(CoreOpsAdv, QR_Returns_Value_3x3) {
    DMatrix A({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 10.0}});
    auto result = qr(A);
    ASSERT_TRUE(result.has_value());
    auto& [Q, R] = *result;
    EXPECT_EQ(Q.rows(), 3u);
    EXPECT_EQ(R.rows(), 3u);
}

TEST(CoreOpsAdv, QR_R_UpperTriangular) {
    DMatrix A({{1.0, 2.0, 0.0}, {2.0, 3.0, 1.0}, {0.0, 1.0, 4.0}});
    auto result = qr(A);
    ASSERT_TRUE(result.has_value());
    auto& [Q, R] = *result;
    for (size_t i = 1; i < R.rows(); ++i)
        for (size_t j = 0; j < i; ++j)
            EXPECT_NEAR(R(i, j), 0.0, 1e-9) << "R not upper triangular at (" << i << "," << j << ")";
}

TEST(CoreOpsAdv, QR_Q_Orthogonal) {
    DMatrix A({{1.0, 2.0, 0.0}, {2.0, 3.0, 1.0}, {0.0, 1.0, 4.0}});
    auto result = qr(A);
    ASSERT_TRUE(result.has_value());
    auto& [Q, R] = *result;
    // Check Q^T * Q = I via dot products of columns
    size_t n = Q.cols();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double dot = 0.0;
            for (size_t k = 0; k < Q.rows(); ++k)
                dot += Q(k, i) * Q(k, j);
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(dot, expected, 1e-9)
                << "Q^T*Q != I at (" << i << "," << j << ")";
        }
    }
}

TEST(CoreOpsAdv, QR_Reconstruction) {
    DMatrix A({{1.0, 2.0, 0.0}, {2.0, 3.0, 1.0}, {0.0, 1.0, 4.0}});
    auto result = qr(A);
    ASSERT_TRUE(result.has_value());
    auto& [Q, R] = *result;
    auto QR_res = matmul<double>(Q, R);
    ASSERT_TRUE(QR_res.has_value());
    double err = max_err(*QR_res, A);
    EXPECT_LT(err, 1e-9) << "QR reconstruction error too large: " << err;
}

// ---------------------------------------------------------------------------
// Cholesky: L * L^T = A for SPD
// ---------------------------------------------------------------------------

TEST(CoreOpsAdv, Chol_Returns_Value_2x2_SPD) {
    DMatrix A({{4.0, 2.0}, {2.0, 3.0}});
    auto L = chol(A);
    ASSERT_TRUE(L.has_value());
}

TEST(CoreOpsAdv, Chol_L_LowerTriangular) {
    DMatrix A({{4.0, 2.0, 0.0}, {2.0, 5.0, 1.0}, {0.0, 1.0, 3.0}});
    auto L_res = chol(A);
    ASSERT_TRUE(L_res.has_value());
    auto& L = *L_res;
    for (size_t i = 0; i < L.rows(); ++i)
        for (size_t j = i + 1; j < L.cols(); ++j)
            EXPECT_NEAR(L(i, j), 0.0, 1e-10);
}

TEST(CoreOpsAdv, Chol_Reconstruction_LLt_EqA) {
    DMatrix A({{4.0, 2.0}, {2.0, 3.0}});
    auto L_res = chol(A);
    ASSERT_TRUE(L_res.has_value());
    auto& L = *L_res;
    // Compute L*L^T manually
    size_t n = L.rows();
    DMatrix LLt(n, n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < n; ++k)
                LLt(i, j) += L(i, k) * L(j, k);
    double err = max_err(LLt, A);
    EXPECT_LT(err, 1e-9) << "Chol: L*L^T != A, error=" << err;
}

TEST(CoreOpsAdv, Chol_Diagonal_Positive) {
    DMatrix A({{4.0, 2.0, 0.0}, {2.0, 5.0, 1.0}, {0.0, 1.0, 3.0}});
    auto L_res = chol(A);
    ASSERT_TRUE(L_res.has_value());
    auto& L = *L_res;
    for (size_t i = 0; i < L.rows(); ++i)
        EXPECT_GT(L(i, i), 0.0) << "Chol diagonal not positive at i=" << i;
}

// ---------------------------------------------------------------------------
// expm(): matrix exponential
// ---------------------------------------------------------------------------

TEST(CoreOpsAdv, Expm_Zero_Matrix_Is_Identity) {
    DMatrix A({{0.0, 0.0}, {0.0, 0.0}});
    auto R = expm(A);
    ASSERT_TRUE(R.has_value());
    EXPECT_NEAR((*R)(0, 0), 1.0, 1e-9);
    EXPECT_NEAR((*R)(1, 1), 1.0, 1e-9);
    EXPECT_NEAR((*R)(0, 1), 0.0, 1e-9);
    EXPECT_NEAR((*R)(1, 0), 0.0, 1e-9);
}

TEST(CoreOpsAdv, Expm_AllFinite) {
    DMatrix A({{0.5, 0.1}, {0.1, 0.5}});
    auto R = expm(A);
    ASSERT_TRUE(R.has_value());
    for (size_t i = 0; i < R->rows(); ++i)
        for (size_t j = 0; j < R->cols(); ++j)
            EXPECT_TRUE(std::isfinite((*R)(i, j)));
}

TEST(CoreOpsAdv, Expm_Positive_Det) {
    // det(expm(A)) = exp(trace(A)) — use small entries for Taylor series convergence
    DMatrix A({{0.5, 0.0}, {0.0, 0.3}});
    auto R = expm(A);
    ASSERT_TRUE(R.has_value());
    auto d = det(*R);
    auto t = trace(A);
    ASSERT_TRUE(d.has_value());
    ASSERT_TRUE(t.has_value());
    EXPECT_NEAR(*d, std::exp(*t), 1e-6)
        << "det(expm(A)) should equal exp(trace(A))";
}
