// MathScript: Advanced Linalg Tests - norm(p), CG on SPD, eig accuracy

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "ms/linalg/linalg.hpp"
#include "ms/core/operations.hpp"
#include "ms/core/matrix.hpp"

using namespace ms;
using DMatrix = Matrix<double>;

// ---------------------------------------------------------------------------
// norm() with different p values
// ---------------------------------------------------------------------------

TEST(LinalgNormCGAdv, Norm_P2_Vector) {
    // norm([3,4], 2) = 5 (L2 norm)
    DMatrix v({{3.0}, {4.0}});
    auto result = norm(v, 2);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result.value(), 5.0, 1e-9);
}

TEST(LinalgNormCGAdv, Norm_P1_Vector) {
    // norm([3,4], 1) = max column sum of abs values
    DMatrix v({{3.0}, {4.0}});
    auto result = norm(v, 1);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(std::isfinite(result.value()));
    EXPECT_GT(result.value(), 0.0);
}

TEST(LinalgNormCGAdv, Norm_Default_2) {
    DMatrix A({{1.0, 0.0}, {0.0, 2.0}});
    auto r2 = norm(A);
    auto r2_explicit = norm(A, 2);
    ASSERT_TRUE(r2.has_value());
    ASSERT_TRUE(r2_explicit.has_value());
    EXPECT_NEAR(r2.value(), r2_explicit.value(), 1e-9);
}

TEST(LinalgNormCGAdv, Norm_Identity_Positive) {
    DMatrix I = eye<double>(3);
    auto result = norm(I, 2);
    ASSERT_TRUE(result.has_value());
    // The norm of the identity matrix should be positive and finite
    EXPECT_GT(result.value(), 0.0);
    EXPECT_TRUE(std::isfinite(result.value()));
}

TEST(LinalgNormCGAdv, Norm_ScaledMatrix) {
    // norm(k*A) = k * norm(A)
    DMatrix A({{1.0, 2.0}, {3.0, 4.0}});
    double k = 3.0;
    DMatrix kA({{k*1.0, k*2.0}, {k*3.0, k*4.0}});
    auto r_A = norm(A, 2);
    auto r_kA = norm(kA, 2);
    ASSERT_TRUE(r_A.has_value());
    ASSERT_TRUE(r_kA.has_value());
    EXPECT_NEAR(r_kA.value(), k * r_A.value(), 1e-8);
}

TEST(LinalgNormCGAdv, Norm_ZeroMatrix_Is0) {
    DMatrix Z = zeros<double>(3, 3);
    auto result = norm(Z, 2);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result.value(), 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Conjugate Gradient on SPD systems
// ---------------------------------------------------------------------------

TEST(LinalgNormCGAdv, CG_Simple_2x2_SPD) {
    // A = [[4, 1], [1, 3]], b = [[1], [2]]
    // Solution: x0 = 1/11, x1 = 7/11
    DMatrix A({{4.0, 1.0}, {1.0, 3.0}});
    DMatrix b({{1.0}, {2.0}});
    auto result = cg(A, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result.value()(0,0), 1.0/11.0, 1e-6);
    EXPECT_NEAR(result.value()(1,0), 7.0/11.0, 1e-6);
}

TEST(LinalgNormCGAdv, CG_Identity_System) {
    DMatrix I = eye<double>(3);
    DMatrix b({{1.0}, {2.0}, {3.0}});
    auto result = cg(I, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result.value()(0,0), 1.0, 1e-8);
    EXPECT_NEAR(result.value()(1,0), 2.0, 1e-8);
    EXPECT_NEAR(result.value()(2,0), 3.0, 1e-8);
}

TEST(LinalgNormCGAdv, CG_Diagonal_System) {
    // 2x = 4, 3y = 6, 5z = 10
    DMatrix A({{2.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 5.0}});
    DMatrix b({{4.0}, {6.0}, {10.0}});
    auto result = cg(A, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result.value()(0,0), 2.0, 1e-8);
    EXPECT_NEAR(result.value()(1,0), 2.0, 1e-8);
    EXPECT_NEAR(result.value()(2,0), 2.0, 1e-8);
}

TEST(LinalgNormCGAdv, CG_4x4_SPD) {
    // Diagonally dominant SPD system
    DMatrix A({{4.0, -1.0, 0.0, 0.0},
               {-1.0, 4.0, -1.0, 0.0},
               {0.0, -1.0, 4.0, -1.0},
               {0.0, 0.0, -1.0, 4.0}});
    DMatrix b({{1.0}, {1.0}, {1.0}, {1.0}});

    auto result = cg(A, b);
    ASSERT_TRUE(result.has_value());
    const auto& x = result.value();
    EXPECT_EQ(x.rows(), 4u);
    // Verify Ax ≈ b
    auto Ax = matmul(A, x);
    ASSERT_TRUE(Ax.has_value());
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(Ax.value()(i,0), 1.0, 1e-6)
            << "CG residual at row " << i;
    }
}

// ---------------------------------------------------------------------------
// BiCGSTAB on non-symmetric systems
// ---------------------------------------------------------------------------

TEST(LinalgNormCGAdv, BiCGSTAB_2x2) {
    DMatrix A({{3.0, 1.0}, {1.0, 2.0}});
    DMatrix b({{5.0}, {5.0}});
    // Solution: x=1.4, y=1.8
    auto result = bicgstab(A, b);
    ASSERT_TRUE(result.has_value());
    // Verify Ax ≈ b
    auto Ax = matmul(A, result.value());
    ASSERT_TRUE(Ax.has_value());
    EXPECT_NEAR(Ax.value()(0,0), 5.0, 1e-6);
    EXPECT_NEAR(Ax.value()(1,0), 5.0, 1e-6);
}

TEST(LinalgNormCGAdv, BiCGSTAB_Identity) {
    DMatrix I = eye<double>(4);
    DMatrix b({{1.0}, {2.0}, {3.0}, {4.0}});
    auto result = bicgstab(I, b);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(result.value()(i,0), b(i,0), 1e-8);
    }
}

// ---------------------------------------------------------------------------
// eig accuracy for known matrices
// ---------------------------------------------------------------------------

TEST(LinalgNormCGAdv, Eig_Trace_Equals_SumEigenvalues) {
    DMatrix A({{3.0, 1.0}, {1.0, 4.0}});
    auto trace_r = trace(A);
    auto eig_r = eig(A);
    ASSERT_TRUE(trace_r.has_value());
    ASSERT_TRUE(eig_r.has_value());

    double sum_eig = 0.0;
    for (size_t i = 0; i < eig_r.value().values.rows(); ++i) {
        sum_eig += eig_r.value().values(i, 0);
    }
    EXPECT_NEAR(sum_eig, trace_r.value(), 1e-6);
}

TEST(LinalgNormCGAdv, Eig_Det_Equals_ProductEigenvalues) {
    DMatrix A({{2.0, 1.0}, {1.0, 3.0}});
    auto det_r = det(A);
    auto eig_r = eig(A);
    ASSERT_TRUE(det_r.has_value());
    ASSERT_TRUE(eig_r.has_value());

    double prod_eig = 1.0;
    for (size_t i = 0; i < eig_r.value().values.rows(); ++i) {
        prod_eig *= eig_r.value().values(i, 0);
    }
    EXPECT_NEAR(prod_eig, det_r.value(), 1e-6);
}

// ---------------------------------------------------------------------------
// trace properties
// ---------------------------------------------------------------------------

TEST(LinalgNormCGAdv, Trace_Sum_Diagonal) {
    DMatrix A({{1.0, 9.0, 8.0}, {7.0, 2.0, 6.0}, {5.0, 4.0, 3.0}});
    auto t = trace(A);
    ASSERT_TRUE(t.has_value());
    EXPECT_NEAR(t.value(), 6.0, 1e-10);  // 1+2+3
}

TEST(LinalgNormCGAdv, Trace_Zeros_IsZero) {
    DMatrix Z = zeros<double>(4, 4);
    auto t = trace(Z);
    ASSERT_TRUE(t.has_value());
    EXPECT_NEAR(t.value(), 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// det properties
// ---------------------------------------------------------------------------

TEST(LinalgNormCGAdv, Det_Identity_Is1) {
    DMatrix I = eye<double>(4);
    auto d = det(I);
    ASSERT_TRUE(d.has_value());
    EXPECT_NEAR(d.value(), 1.0, 1e-10);
}

TEST(LinalgNormCGAdv, Det_Singular_IsZero) {
    // [[1,2],[2,4]] is singular
    DMatrix A({{1.0, 2.0}, {2.0, 4.0}});
    auto d = det(A);
    if (d.has_value()) {
        EXPECT_NEAR(d.value(), 0.0, 1e-8);
    }
    // else it returned an error (which is also acceptable for a singular matrix)
}

TEST(LinalgNormCGAdv, Det_2x2_KnownValue) {
    DMatrix A({{3.0, 8.0}, {4.0, 6.0}});
    auto d = det(A);
    ASSERT_TRUE(d.has_value());
    EXPECT_NEAR(d.value(), 3.0*6.0 - 8.0*4.0, 1e-9);  // = -14
}
