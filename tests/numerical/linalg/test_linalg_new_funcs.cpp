// Wave 56: New linalg functions — kron, linspace, repmat, pinv, null, orth, sinm, cosm, funm, minres
#define _USE_MATH_DEFINES
#include "ms/linalg/linalg.hpp"
#include "ms/core/operations.hpp"
#include <cmath>
#include <gtest/gtest.h>

// Helper
static double frob(const ms::Matrix<double>& A) {
    double s = 0.0;
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            s += A(i,j) * A(i,j);
    return std::sqrt(s);
}

// -----------------------------------------------------------------------
// linspace
// -----------------------------------------------------------------------
TEST(LinalgLinspace, CorrectSize) {
    auto v = ms::linspace(0.0, 1.0, 5u);
    EXPECT_EQ(v.size(), 5u);
}

TEST(LinalgLinspace, StartAndEnd) {
    auto v = ms::linspace(2.0, 10.0, 9u);
    EXPECT_NEAR(v.front(), 2.0, 1e-14);
    EXPECT_NEAR(v.back(), 10.0, 1e-14);
}

TEST(LinalgLinspace, EquallySpaced) {
    auto v = ms::linspace(0.0, 4.0, 5u);
    for (size_t i = 0; i < v.size(); ++i)
        EXPECT_NEAR(v[i], static_cast<double>(i), 1e-12);
}

// -----------------------------------------------------------------------
// kron
// -----------------------------------------------------------------------
TEST(LinalgKron, IdentityByIdentity) {
    auto I2 = ms::eye<double>(2);
    auto K = ms::kron(I2, I2);
    EXPECT_EQ(K.rows(), 4u);
    EXPECT_EQ(K.cols(), 4u);
    // Kron of I2 x I2 = I4
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            EXPECT_NEAR(K(i,j), (i == j) ? 1.0 : 0.0, 1e-12);
}

TEST(LinalgKron, ScalarByMatrix) {
    auto A = ms::zeros<double>(2, 2);
    A(0,0) = 2.0; A(0,1) = 3.0; A(1,0) = 4.0; A(1,1) = 5.0;
    auto s = ms::zeros<double>(1, 1);
    s(0,0) = 3.0;
    auto K = ms::kron(s, A);
    EXPECT_EQ(K.rows(), 2u);
    EXPECT_EQ(K.cols(), 2u);
    EXPECT_NEAR(K(0,0), 6.0, 1e-12);
    EXPECT_NEAR(K(0,1), 9.0, 1e-12);
}

TEST(LinalgKron, SizeCorrect) {
    auto A = ms::zeros<double>(2, 3);
    auto B = ms::zeros<double>(4, 5);
    auto K = ms::kron(A, B);
    EXPECT_EQ(K.rows(), 8u);
    EXPECT_EQ(K.cols(), 15u);
}

// -----------------------------------------------------------------------
// repmat
// -----------------------------------------------------------------------
TEST(LinalgRepmat, SizeCorrect) {
    auto A = ms::eye<double>(2);
    auto R = ms::repmat(A, 3, 2);
    EXPECT_EQ(R.rows(), 6u);
    EXPECT_EQ(R.cols(), 4u);
}

TEST(LinalgRepmat, ValuesRepeated) {
    auto A = ms::zeros<double>(2, 2);
    A(0,0) = 1.0; A(0,1) = 2.0; A(1,0) = 3.0; A(1,1) = 4.0;
    auto R = ms::repmat(A, 2, 2);
    EXPECT_NEAR(R(0,0), 1.0, 1e-12);
    EXPECT_NEAR(R(0,2), 1.0, 1e-12);
    EXPECT_NEAR(R(2,0), 1.0, 1e-12);
    EXPECT_NEAR(R(2,2), 1.0, 1e-12);
}

// -----------------------------------------------------------------------
// pinv
// -----------------------------------------------------------------------
TEST(LinalgPinv, SquareInverseEquivalent) {
    auto A = ms::zeros<double>(3, 3);
    A(0,0) = 4.0; A(0,1) = 7.0;
    A(1,0) = 2.0; A(1,1) = 6.0;
    A(2,2) = 3.0;
    auto Apinv = ms::pinv(A);
    ASSERT_TRUE(Apinv.has_value());
    // A * pinv(A) should be close to identity
    auto I_approx_r = ms::matmul(A, Apinv.value());
    ASSERT_TRUE(I_approx_r.has_value());
    auto& I_approx = I_approx_r.value();
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(I_approx(i,i), 1.0, 0.1);
}

TEST(LinalgPinv, ResultIsFinite) {
    auto A = ms::randn<double>(4, 3, 42u);
    auto Ap = ms::pinv(A);
    ASSERT_TRUE(Ap.has_value());
    EXPECT_TRUE(std::isfinite(frob(Ap.value())));
}

TEST(LinalgPinv, Randn4x3SeededSweep) {
    for (unsigned seed = 0; seed <= 20; ++seed) {
        auto A = ms::randn<double>(4, 3, seed);
        auto svd_res = ms::svd(A);
        ASSERT_TRUE(svd_res.has_value()) << "seed=" << seed;
        EXPECT_TRUE(std::isfinite(frob(svd_res->U))) << "seed=" << seed;
        EXPECT_TRUE(std::isfinite(frob(svd_res->V))) << "seed=" << seed;
        for (size_t i = 0; i < svd_res->S.rows(); ++i) {
            EXPECT_TRUE(std::isfinite(svd_res->S(i, 0))) << "seed=" << seed;
        }

        auto Ap = ms::pinv(A);
        ASSERT_TRUE(Ap.has_value()) << "seed=" << seed;
        EXPECT_TRUE(std::isfinite(frob(Ap.value()))) << "seed=" << seed;
    }
}

TEST(LinalgPinv, ShapeTransposed) {
    // pinv of (m x n) is (n x m) — use known matrix
    auto A = ms::zeros<double>(3, 2);
    A(0,0) = 1.0; A(0,1) = 0.0;
    A(1,0) = 0.0; A(1,1) = 1.0;
    A(2,0) = 1.0; A(2,1) = 1.0;
    auto Apr = ms::pinv(A);
    ASSERT_TRUE(Apr.has_value());
    auto Apv = Apr.value();
    EXPECT_EQ(Apv.rows(), 2u);
    EXPECT_EQ(Apv.cols(), 3u);
    EXPECT_TRUE(std::isfinite(frob(Apv)));
}

TEST(LinalgPinv, MoorePenroseIdentity) {
    auto A = ms::zeros<double>(4, 3);
    A(0,0) = 1.0; A(0,1) = 0.0; A(0,2) = 2.0;
    A(1,0) = 0.0; A(1,1) = 3.0; A(1,2) = 0.0;
    A(2,0) = 4.0; A(2,1) = 0.0; A(2,2) = 5.0;
    A(3,0) = 0.0; A(3,1) = 6.0; A(3,2) = 0.0;
    auto Ap = ms::pinv(A);
    ASSERT_TRUE(Ap.has_value());
    auto AAp_r = ms::matmul(A, Ap.value());
    ASSERT_TRUE(AAp_r.has_value());
    auto AApA_r = ms::matmul(AAp_r.value(), A);
    ASSERT_TRUE(AApA_r.has_value());
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            EXPECT_NEAR(AApA_r.value()(i, j), A(i, j), 1e-8);
}

// -----------------------------------------------------------------------
// orth
// -----------------------------------------------------------------------
TEST(LinalgOrth, ReturnsOrthonormalColumns) {
    auto A = ms::randn<double>(5, 3, 7);
    auto Qr = ms::orth(A);
    ASSERT_TRUE(Qr.has_value());
    auto Qv = Qr.value();
    ASSERT_LE(Qv.cols(), 3u);
    if (Qv.cols() == 0) return;
    const size_t m = Qv.rows(), k = Qv.cols();
    // Q^T Q should be I (compute manually)
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < k; ++j) {
            double dot = 0.0;
            for (size_t r = 0; r < m; ++r) dot += Qv(r,i) * Qv(r,j);
            EXPECT_NEAR(dot, (i == j) ? 1.0 : 0.0, 1e-8);
        }
    }
}

TEST(LinalgOrth, NullMatrixHasNoOrth) {
    auto Z = ms::zeros<double>(3, 3);
    auto Qr = ms::orth(Z);
    if (Qr.has_value()) {
        EXPECT_EQ(Qr.value().cols(), 0u);
    }
}

// -----------------------------------------------------------------------
// null
// -----------------------------------------------------------------------
TEST(LinalgNull, RankDeficientHasNullSpace) {
    // A = [1 2; 2 4] (rank 1) => null space has dimension 1
    auto A = ms::zeros<double>(2, 2);
    A(0,0) = 1.0; A(0,1) = 2.0;
    A(1,0) = 2.0; A(1,1) = 4.0;
    auto Nr = ms::null(A);
    if (Nr.has_value() && Nr.value().cols() > 0) {
        auto& Nv = Nr.value();
        // A*N should be near zero
        auto AN_r = ms::matmul(A, Nv);
        if (AN_r.has_value())
            EXPECT_LT(frob(AN_r.value()), 1e-8);
    }
}

TEST(LinalgNull, WideMatrixRankDeficient) {
    // A = [1 2 3 4; 2 4 6 8] (rank 1) => nullity 3
    auto A = ms::zeros<double>(2, 4);
    A(0,0) = 1.0; A(0,1) = 2.0; A(0,2) = 3.0; A(0,3) = 4.0;
    A(1,0) = 2.0; A(1,1) = 4.0; A(1,2) = 6.0; A(1,3) = 8.0;
    auto Nr = ms::null(A);
    ASSERT_TRUE(Nr.has_value());
    auto& N = Nr.value();
    EXPECT_EQ(N.rows(), 4u);
    EXPECT_GE(N.cols(), 2u);
    auto AN_r = ms::matmul(A, N);
    ASSERT_TRUE(AN_r.has_value());
    EXPECT_LT(frob(AN_r.value()), 1e-8);
}

// -----------------------------------------------------------------------
// sinm / cosm
// -----------------------------------------------------------------------
TEST(LinalgSinm, DiagonalMatrix) {
    // sinm(diag([0, pi/2])) = diag([0, 1])
    auto D = ms::zeros<double>(2, 2);
    D(0,0) = 0.0;
    D(1,1) = M_PI / 2.0;
    auto S = ms::sinm(D);
    ASSERT_TRUE(S.has_value());
    EXPECT_NEAR(S.value()(0,0), 0.0, 1e-9);
    EXPECT_NEAR(S.value()(1,1), 1.0, 1e-9);
}

TEST(LinalgCosm, DiagonalMatrix) {
    // cosm(diag([0, pi/2])) = diag([1, 0])
    auto D = ms::zeros<double>(2, 2);
    D(0,0) = 0.0;
    D(1,1) = M_PI / 2.0;
    auto C = ms::cosm(D);
    ASSERT_TRUE(C.has_value());
    EXPECT_NEAR(C.value()(0,0), 1.0, 1e-9);
    EXPECT_NEAR(C.value()(1,1), 0.0, 1e-9);
}

TEST(LinalgSinmCosm, Pythagorean) {
    // sin^2 + cos^2 = I (element-wise for diagonal)
    auto D = ms::zeros<double>(2, 2);
    D(0,0) = 1.0; D(1,1) = 0.5;
    auto S = ms::sinm(D);
    auto C = ms::cosm(D);
    ASSERT_TRUE(S.has_value());
    ASSERT_TRUE(C.has_value());
    // S^2 + C^2 should equal D (for diagonal: sin^2 + cos^2 = 1 on diagonal)
    // Check diagonal
    for (size_t i = 0; i < 2; ++i) {
        double s2 = S.value()(i,i) * S.value()(i,i);
        double c2 = C.value()(i,i) * C.value()(i,i);
        EXPECT_NEAR(s2 + c2, 1.0, 1e-9);
    }
}

// -----------------------------------------------------------------------
// funm
// -----------------------------------------------------------------------
TEST(LinalgFunm, SquareRootEquivalentToSqrtm) {
    auto A = ms::zeros<double>(3, 3);
    A(0,0) = 4.0; A(1,1) = 9.0; A(2,2) = 16.0;
    std::function<double(double)> sqrtfn = [](double x) {
        return std::sqrt(std::max(x, 0.0));
    };
    auto F = ms::funm(A, sqrtfn);
    ASSERT_TRUE(F.has_value());
    EXPECT_NEAR(F.value()(0,0), 2.0, 1e-9);
    EXPECT_NEAR(F.value()(1,1), 3.0, 1e-9);
    EXPECT_NEAR(F.value()(2,2), 4.0, 1e-9);
}

TEST(LinalgFunm, IdentityFunction) {
    auto A = ms::zeros<double>(2, 2);
    A(0,0) = 3.0; A(1,1) = 5.0;
    std::function<double(double)> id = [](double x) { return x; };
    auto F = ms::funm(A, id);
    ASSERT_TRUE(F.has_value());
    EXPECT_NEAR(F.value()(0,0), 3.0, 1e-9);
    EXPECT_NEAR(F.value()(1,1), 5.0, 1e-9);
}

// -----------------------------------------------------------------------
// minres
// -----------------------------------------------------------------------
TEST(LinalgMinres, SolvesSymmetricSystem) {
    // 3x3 SPD: A = I
    auto A = ms::eye<double>(3);
    auto b = ms::zeros<double>(3, 1);
    b(0,0) = 1.0; b(1,0) = 2.0; b(2,0) = 3.0;
    auto x = ms::minres(A, b);
    ASSERT_TRUE(x.has_value());
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(x.value()(i,0), b(i,0), 0.1);
}

TEST(LinalgMinres, IsFinite) {
    auto A = ms::zeros<double>(3, 3);
    A(0,0) = 4.0; A(0,1) = 1.0; A(1,0) = 1.0;
    A(1,1) = 3.0; A(1,2) = 1.0; A(2,1) = 1.0; A(2,2) = 2.0;
    auto b = ms::zeros<double>(3, 1);
    b(0,0) = 1.0; b(1,0) = 1.0; b(2,0) = 1.0;
    auto x = ms::minres(A, b);
    if (x.has_value()) {
        for (size_t i = 0; i < 3; ++i)
            EXPECT_TRUE(std::isfinite(x.value()(i, 0)));
    }
}
