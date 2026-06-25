// MathScript Advanced Iterative Solver Tests (Wave 51)
// CG, BICGSTAB, GMRES convergence and accuracy tests.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

#include "ms/linalg/linalg.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = Matrix<double>;

// Helper: build a symmetric positive definite matrix
static DMatrix make_spd(size_t n, double diag_add = 10.0) {
    DMatrix A(n, n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            A(i, j) = (i == j) ? (static_cast<double>(n) + diag_add) : 1.0;
    return A;
}

// Helper: compute Euclidean norm of a column vector
static double vec_norm(const DMatrix& v) {
    double s = 0.0;
    for (size_t i = 0; i < v.rows(); ++i)
        s += v(i, 0) * v(i, 0);
    return std::sqrt(s);
}

// Helper: matvec A*x (manual, avoids RowMajor linker issues)
static DMatrix matvec_manual(const DMatrix& A, const DMatrix& x) {
    size_t n = A.rows();
    DMatrix r(n, 1);
    for (size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < A.cols(); ++j) s += A(i, j) * x(j, 0);
        r(i, 0) = s;
    }
    return r;
}

// Helper: build column vector from std::initializer_list
static DMatrix col_vec(std::initializer_list<double> vals) {
    DMatrix v(vals.size(), 1);
    size_t i = 0;
    for (double d : vals) v(i++, 0) = d;
    return v;
}

// ---------------------------------------------------------------------------
// CG: Conjugate Gradient
// ---------------------------------------------------------------------------

TEST(LinalgIterAdv, CG_SPD3x3_Converges) {
    DMatrix A = make_spd(3);
    DMatrix b = col_vec({1.0, 2.0, 3.0});
    auto result = cg(A, b, 100, 1e-10);
    ASSERT_TRUE(result.has_value()) << "CG failed to converge";
    auto& x = result.value();
    // Check residual ||Ax - b|| / ||b||
    auto r = matvec_manual(A, x);
    double res = 0.0;
    for (size_t i = 0; i < b.rows(); ++i)
        res += (r(i, 0) - b(i, 0)) * (r(i, 0) - b(i, 0));
    res = std::sqrt(res);
    EXPECT_LT(res, 1e-6) << "CG residual too large";
}

TEST(LinalgIterAdv, CG_SPD5x5_Converges) {
    DMatrix A = make_spd(5);
    DMatrix b = col_vec({1.0, 0.0, 2.0, 0.0, 1.0});
    auto result = cg(A, b, 500, 1e-10);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    auto r = matvec_manual(A, x);
    double res = 0.0;
    for (size_t i = 0; i < b.rows(); ++i)
        res += (r(i, 0) - b(i, 0)) * (r(i, 0) - b(i, 0));
    EXPECT_LT(std::sqrt(res), 1e-6);
}

TEST(LinalgIterAdv, CG_IdentityMatrix_ImmediateConvergence) {
    // A = I, b = any => x = b
    size_t n = 4;
    DMatrix A(n, n);
    for (size_t i = 0; i < n; ++i) A(i, i) = 1.0;
    DMatrix b = col_vec({1.0, 2.0, 3.0, 4.0});
    auto result = cg(A, b, 100, 1e-12);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(x(i, 0), b(i, 0), 1e-8);
}

TEST(LinalgIterAdv, CG_MatchesDirectSolve) {
    DMatrix A = make_spd(4, 20.0);
    DMatrix b = col_vec({3.0, 1.0, 4.0, 1.0});
    auto cg_res    = cg(A, b, 500, 1e-12);
    auto dir_res   = solve(A, b);
    ASSERT_TRUE(cg_res.has_value());
    ASSERT_TRUE(dir_res.has_value());
    auto& xc = cg_res.value();
    auto& xd = dir_res.value();
    for (size_t i = 0; i < b.rows(); ++i)
        EXPECT_NEAR(xc(i, 0), xd(i, 0), 1e-6)
            << "CG solution differs from direct solve at row " << i;
}

TEST(LinalgIterAdv, CG_SolutionFinite) {
    DMatrix A = make_spd(3);
    DMatrix b = col_vec({5.0, -1.0, 2.0});
    auto result = cg(A, b, 100, 1e-10);
    if (result.has_value()) {
        auto& x = result.value();
        for (size_t i = 0; i < x.rows(); ++i)
            EXPECT_TRUE(std::isfinite(x(i, 0)));
    }
}

// ---------------------------------------------------------------------------
// BICGSTAB: Bi-Conjugate Gradient Stabilized
// ---------------------------------------------------------------------------

TEST(LinalgIterAdv, BICGSTAB_SPD3x3_Converges) {
    DMatrix A = make_spd(3);
    DMatrix b = col_vec({2.0, 3.0, 1.0});
    auto result = bicgstab(A, b, 200, 1e-10);
    ASSERT_TRUE(result.has_value()) << "BICGSTAB failed";
    auto& x = result.value();
    auto r = matvec_manual(A, x);
    double res = 0.0;
    for (size_t i = 0; i < b.rows(); ++i)
        res += (r(i, 0) - b(i, 0)) * (r(i, 0) - b(i, 0));
    EXPECT_LT(std::sqrt(res), 1e-5);
}

TEST(LinalgIterAdv, BICGSTAB_IdentityMatrix_SolutionIsRHS) {
    size_t n = 3;
    DMatrix A(n, n);
    for (size_t i = 0; i < n; ++i) A(i, i) = 1.0;
    DMatrix b = col_vec({7.0, -2.0, 5.0});
    auto result = bicgstab(A, b, 100, 1e-12);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(x(i, 0), b(i, 0), 1e-8);
}

TEST(LinalgIterAdv, BICGSTAB_DiagonalMatrix_SolutionKnown) {
    // A = diag(2, 4, 8), b = (2, 4, 8) => x = (1, 1, 1)
    size_t n = 3;
    DMatrix A(n, n);
    A(0, 0) = 2.0; A(1, 1) = 4.0; A(2, 2) = 8.0;
    DMatrix b = col_vec({2.0, 4.0, 8.0});
    auto result = bicgstab(A, b, 100, 1e-12);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(x(i, 0), 1.0, 1e-8);
}

TEST(LinalgIterAdv, BICGSTAB_MatchesCG_OnSPD) {
    DMatrix A = make_spd(3, 15.0);
    DMatrix b = col_vec({1.0, 2.0, 3.0});
    auto cg_res    = cg(A, b, 500, 1e-12);
    auto bcg_res   = bicgstab(A, b, 500, 1e-12);
    ASSERT_TRUE(cg_res.has_value());
    ASSERT_TRUE(bcg_res.has_value());
    auto& xc = cg_res.value();
    auto& xb = bcg_res.value();
    for (size_t i = 0; i < b.rows(); ++i)
        EXPECT_NEAR(xc(i, 0), xb(i, 0), 1e-5)
            << "CG and BICGSTAB disagree at row " << i;
}

TEST(LinalgIterAdv, BICGSTAB_SolutionFinite) {
    DMatrix A = make_spd(4, 8.0);
    DMatrix b = col_vec({1.0, 1.0, 1.0, 1.0});
    auto result = bicgstab(A, b, 500, 1e-10);
    if (result.has_value()) {
        auto& x = result.value();
        for (size_t i = 0; i < x.rows(); ++i)
            EXPECT_TRUE(std::isfinite(x(i, 0)));
    }
}

// ---------------------------------------------------------------------------
// GMRES: Generalized Minimal Residual
// ---------------------------------------------------------------------------

TEST(LinalgIterAdv, GMRES_SPD3x3_Converges) {
    DMatrix A = make_spd(3);
    DMatrix b = col_vec({1.0, 1.0, 1.0});
    auto result = gmres(A, b, 20, 200, 1e-10);
    ASSERT_TRUE(result.has_value()) << "GMRES failed";
    auto& x = result.value();
    auto r = matvec_manual(A, x);
    double res = 0.0;
    for (size_t i = 0; i < b.rows(); ++i)
        res += (r(i, 0) - b(i, 0)) * (r(i, 0) - b(i, 0));
    EXPECT_LT(std::sqrt(res), 1e-5);
}

TEST(LinalgIterAdv, GMRES_IdentityMatrix_SolutionIsRHS) {
    size_t n = 4;
    DMatrix A(n, n);
    for (size_t i = 0; i < n; ++i) A(i, i) = 1.0;
    DMatrix b = col_vec({3.0, 1.0, 4.0, 1.0});
    auto result = gmres(A, b, 10, 100, 1e-12);
    ASSERT_TRUE(result.has_value());
    auto& x = result.value();
    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(x(i, 0), b(i, 0), 1e-8);
}

TEST(LinalgIterAdv, GMRES_MatchesDirectSolve) {
    DMatrix A = make_spd(3, 12.0);
    DMatrix b = col_vec({2.0, -1.0, 3.0});
    auto gmres_res = gmres(A, b, 10, 200, 1e-12);
    auto dir_res   = solve(A, b);
    ASSERT_TRUE(gmres_res.has_value());
    ASSERT_TRUE(dir_res.has_value());
    auto& xg = gmres_res.value();
    auto& xd = dir_res.value();
    for (size_t i = 0; i < b.rows(); ++i)
        EXPECT_NEAR(xg(i, 0), xd(i, 0), 1e-5);
}

TEST(LinalgIterAdv, GMRES_SolutionFinite) {
    DMatrix A = make_spd(4);
    DMatrix b = col_vec({1.0, 2.0, 3.0, 4.0});
    auto result = gmres(A, b, 20, 200, 1e-10);
    if (result.has_value()) {
        auto& x = result.value();
        for (size_t i = 0; i < x.rows(); ++i)
            EXPECT_TRUE(std::isfinite(x(i, 0)));
    }
}

// ---------------------------------------------------------------------------
// All three solvers agree on same SPD system
// ---------------------------------------------------------------------------

TEST(LinalgIterAdv, AllSolvers_AgreeOnSPD4x4) {
    DMatrix A = make_spd(4, 20.0);
    DMatrix b = col_vec({1.0, 2.0, 3.0, 4.0});
    auto cg_res    = cg(A, b, 500, 1e-12);
    auto bcg_res   = bicgstab(A, b, 500, 1e-12);
    auto gm_res    = gmres(A, b, 20, 200, 1e-12);
    ASSERT_TRUE(cg_res.has_value());
    ASSERT_TRUE(bcg_res.has_value());
    ASSERT_TRUE(gm_res.has_value());
    for (size_t i = 0; i < b.rows(); ++i) {
        EXPECT_NEAR(cg_res.value()(i, 0), bcg_res.value()(i, 0), 1e-5);
        EXPECT_NEAR(cg_res.value()(i, 0), gm_res.value()(i, 0), 1e-5);
    }
}
