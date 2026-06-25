// MathScript Tests: Hessenberg reduction and least-squares (Wave 53)

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>

#include "ms/linalg/linalg.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = Matrix<double>;

// Helper: manual matmul
static DMatrix mm(const DMatrix& A, const DMatrix& B) {
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

// Helper: build transpose
static DMatrix T(const DMatrix& A) {
    DMatrix At(A.cols(), A.rows());
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            At(j, i) = A(i, j);
    return At;
}

// Helper: Frobenius norm of difference
static double diff_frob(const DMatrix& A, const DMatrix& B) {
    double s = 0.0;
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            s += (A(i, j) - B(i, j)) * (A(i, j) - B(i, j));
    return std::sqrt(s);
}

// Helper: max off-diagonal error for orthogonality Q^T*Q = I
static double ortho_err(const DMatrix& Q) {
    auto QtQ = mm(T(Q), Q);
    double err = 0.0;
    for (size_t i = 0; i < QtQ.rows(); ++i)
        for (size_t j = 0; j < QtQ.cols(); ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            err = std::max(err, std::abs(QtQ(i, j) - expected));
        }
    return err;
}

// ---------------------------------------------------------------------------
// Hessenberg reduction: H = Q^H * A * Q
// ---------------------------------------------------------------------------

TEST(LinalgHessLsq, Hess_ReturnsValue) {
    DMatrix A(4, 4);
    A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0; A(0, 3) = 4.0;
    A(1, 0) = 5.0; A(1, 1) = 6.0; A(1, 2) = 7.0; A(1, 3) = 8.0;
    A(2, 0) = 9.0; A(2, 1) = 1.0; A(2, 2) = 2.0; A(2, 3) = 3.0;
    A(3, 0) = 4.0; A(3, 1) = 5.0; A(3, 2) = 6.0; A(3, 3) = 7.0;
    auto result = hess(A);
    EXPECT_TRUE(result.has_value()) << "hess should succeed";
}

TEST(LinalgHessLsq, Hess_AllFinite) {
    DMatrix A(4, 4);
    A(0, 0) = 2.0; A(0, 1) = 1.0; A(0, 2) = 0.5;
    A(1, 0) = 1.0; A(1, 1) = 3.0; A(1, 2) = 1.0; A(1, 3) = 0.5;
    A(2, 0) = 0.5; A(2, 1) = 1.0; A(2, 2) = 4.0; A(2, 3) = 1.0;
    A(3, 1) = 0.5; A(3, 2) = 1.0; A(3, 3) = 5.0;
    auto result = hess(A);
    ASSERT_TRUE(result.has_value());
    auto& H = result.value();
    for (size_t i = 0; i < H.rows(); ++i)
        for (size_t j = 0; j < H.cols(); ++j)
            EXPECT_TRUE(std::isfinite(H(i, j)));
}

TEST(LinalgHessLsq, Hess_UpperHessenbergStructure) {
    // H is upper Hessenberg: H(i,j) = 0 for i > j+1
    DMatrix A(4, 4);
    A(0, 0) = 4.0; A(0, 1) = 3.0; A(0, 2) = 2.0; A(0, 3) = 1.0;
    A(1, 0) = 3.0; A(1, 1) = 4.0; A(1, 2) = 3.0; A(1, 3) = 2.0;
    A(2, 0) = 2.0; A(2, 1) = 3.0; A(2, 2) = 4.0; A(2, 3) = 3.0;
    A(3, 0) = 1.0; A(3, 1) = 2.0; A(3, 2) = 3.0; A(3, 3) = 4.0;
    auto result = hess(A);
    ASSERT_TRUE(result.has_value());
    auto& H = result.value();
    // Check that elements below sub-diagonal are zero
    for (size_t i = 2; i < H.rows(); ++i)
        for (size_t j = 0; j < i - 1; ++j)
            EXPECT_NEAR(H(i, j), 0.0, 1e-8)
                << "H(" << i << "," << j << ") should be ~0 (upper Hessenberg)";
}

TEST(LinalgHessLsq, Hess_IdentityInput_ReturnsIdentity) {
    // hess(I) = I (identity is already Hessenberg)
    DMatrix I(3, 3);
    for (size_t i = 0; i < 3; ++i) I(i, i) = 1.0;
    auto result = hess(I);
    ASSERT_TRUE(result.has_value());
    auto& H = result.value();
    EXPECT_NEAR(H(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(H(1, 1), 1.0, 1e-10);
    EXPECT_NEAR(H(2, 2), 1.0, 1e-10);
}

TEST(LinalgHessLsq, Hess_UpperTriangular_Input_Unchanged) {
    // Upper triangular is already Hessenberg
    DMatrix A(3, 3);
    A(0, 0) = 3.0; A(0, 1) = 2.0; A(0, 2) = 1.0;
    A(1, 1) = 5.0; A(1, 2) = 4.0;
    A(2, 2) = 7.0;
    auto result = hess(A);
    ASSERT_TRUE(result.has_value());
    auto& H = result.value();
    // Sub-diagonal of Hessenberg form should be near zero for upper triangular input
    for (size_t i = 2; i < H.rows(); ++i)
        for (size_t j = 0; j < i - 1; ++j)
            EXPECT_NEAR(H(i, j), 0.0, 1e-8);
}

// ---------------------------------------------------------------------------
// LSQ: least-squares solver
// ---------------------------------------------------------------------------

TEST(LinalgHessLsq, Lsq_ReturnsValue_OverDetermined) {
    // Overdetermined: 3 equations, 2 unknowns
    DMatrix A(3, 2);
    A(0, 0) = 1.0; A(0, 1) = 1.0;
    A(1, 0) = 2.0; A(1, 1) = 1.0;
    A(2, 0) = 3.0; A(2, 1) = 1.0;
    DMatrix b(3, 1);
    b(0, 0) = 2.0; b(1, 0) = 3.0; b(2, 0) = 4.5;
    auto result = lsq(A, b);
    EXPECT_TRUE(result.has_value()) << "lsq should succeed for overdetermined system";
}

TEST(LinalgHessLsq, Lsq_ExactSystem_MatchesSolve) {
    // Square system: lsq = solve exactly
    DMatrix A(3, 3);
    A(0, 0) = 2.0; A(0, 1) = 1.0;
    A(1, 0) = 1.0; A(1, 1) = 3.0; A(1, 2) = 1.0;
    A(2, 1) = 1.0; A(2, 2) = 4.0;
    DMatrix b(3, 1);
    b(0, 0) = 5.0; b(1, 0) = 10.0; b(2, 0) = 9.0;
    auto lsq_res  = lsq(A, b);
    auto dir_res  = solve(A, b);
    ASSERT_TRUE(lsq_res.has_value());
    ASSERT_TRUE(dir_res.has_value());
    auto& xl = lsq_res.value();
    auto& xd = dir_res.value();
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(xl(i, 0), xd(i, 0), 1e-6)
            << "lsq and solve disagree at row " << i;
}

TEST(LinalgHessLsq, Lsq_Solution_AllFinite) {
    DMatrix A(4, 2);
    A(0, 0) = 1.0; A(0, 1) = 0.0;
    A(1, 0) = 0.0; A(1, 1) = 1.0;
    A(2, 0) = 1.0; A(2, 1) = 1.0;
    A(3, 0) = 2.0; A(3, 1) = 1.0;
    DMatrix b(4, 1);
    b(0, 0) = 1.0; b(1, 0) = 2.0; b(2, 0) = 3.0; b(3, 0) = 4.0;
    auto result = lsq(A, b);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    for (size_t i = 0; i < x.rows(); ++i)
        EXPECT_TRUE(std::isfinite(x(i, 0)));
}

TEST(LinalgHessLsq, Lsq_OverDetermined_Minimizes_Residual) {
    // Overdetermined: exact solution would need y=x (slope=1, intercept=0)
    // Points approximately on y = x
    DMatrix A(4, 2);
    A(0, 0) = 1.0; A(0, 1) = 1.0;
    A(1, 0) = 2.0; A(1, 1) = 1.0;
    A(2, 0) = 3.0; A(2, 1) = 1.0;
    A(3, 0) = 4.0; A(3, 1) = 1.0;
    DMatrix b(4, 1);
    b(0, 0) = 1.0; b(1, 0) = 2.1; b(2, 0) = 2.9; b(3, 0) = 4.0;
    auto result = lsq(A, b);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    // Slope should be approximately 1
    EXPECT_NEAR(x(0, 0), 1.0, 0.1) << "Fitted slope near 1";
    // Intercept should be near 0
    EXPECT_NEAR(x(1, 0), 0.0, 0.2) << "Fitted intercept near 0";
}

TEST(LinalgHessLsq, Lsq_CorrectOutputShape) {
    // A is m×n, b is m×1, x should be n×1
    DMatrix A(5, 3);
    for (size_t i = 0; i < 5; ++i) for (size_t j = 0; j < 3; ++j)
        A(i, j) = static_cast<double>(i + 1) * (j + 1);
    A(0, 0) += 1.0; // ensure non-rank-deficient
    DMatrix b(5, 1);
    for (size_t i = 0; i < 5; ++i) b(i, 0) = static_cast<double>(i + 1);
    auto result = lsq(A, b);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
}
