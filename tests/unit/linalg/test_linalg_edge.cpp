#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "ms/linalg/linalg.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = ms::ColMatrix<double>;

// ---- 1x1 matrix operations ----

TEST(LinalgEdgeTest, one_by_one_det) {
    DMatrix A(1, 1);
    A(0, 0) = 7.0;
    const auto d = det(A);
    ASSERT_TRUE(d.has_value());
    EXPECT_NEAR(*d, 7.0, 1e-12);
}

TEST(LinalgEdgeTest, one_by_one_solve) {
    DMatrix A(1, 1);
    A(0, 0) = 3.0;
    DMatrix b(1, 1);
    b(0, 0) = 9.0;
    const auto x = solve(A, b);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)(0, 0), 3.0, 1e-12);
}

TEST(LinalgEdgeTest, one_by_one_inv_via_solve) {
    DMatrix A(1, 1);
    A(0, 0) = 2.0;
    DMatrix I = eye<double>(1);
    const auto Ainv = solve(A, I);
    ASSERT_TRUE(Ainv.has_value());
    EXPECT_NEAR((*Ainv)(0, 0), 0.5, 1e-12);
}

// ---- Large matrix construction ----

TEST(LinalgEdgeTest, zeros_100x100_shape) {
    const DMatrix Z = zeros<double>(100, 100);
    EXPECT_EQ(Z.rows(), 100u);
    EXPECT_EQ(Z.cols(), 100u);
    for (size_t i = 0; i < 100; ++i) {
        for (size_t j = 0; j < 100; ++j) {
            EXPECT_DOUBLE_EQ(Z(i, j), 0.0);
        }
    }
}

TEST(LinalgEdgeTest, eye_50_trace_equals_n) {
    const DMatrix I = eye<double>(50);
    EXPECT_EQ(I.rows(), 50u);
    EXPECT_EQ(I.cols(), 50u);
    const auto t = trace(I);
    ASSERT_TRUE(t.has_value());
    EXPECT_NEAR(*t, 50.0, 1e-12);
}

// ---- SVD of rank-deficient matrix ----

TEST(LinalgEdgeTest, svd_rank_deficient) {
    // Rank-1 matrix: A = [[1,2],[2,4]], singular values = 5 and 0
    DMatrix A{{1.0, 2.0}, {2.0, 4.0}};
    const auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->S.rows(), 2u);
    EXPECT_NEAR(result->S(0, 0), 5.0, 1e-6);
    EXPECT_NEAR(result->S(1, 0), 0.0, 1e-6);
}

// ---- Norm variants ----

TEST(LinalgEdgeTest, norm_vector_p1) {
    // Column vector [3, -4]: 1-norm = |3| + |-4| = 7
    DMatrix v(2, 1);
    v(0, 0) = 3.0;
    v(1, 0) = -4.0;
    const auto n = norm(v, 1);
    ASSERT_TRUE(n.has_value());
    EXPECT_NEAR(*n, 7.0, 1e-12);
}

TEST(LinalgEdgeTest, norm_vector_p2) {
    // Column vector [3, -4]: 2-norm = sqrt(9+16) = 5
    DMatrix v(2, 1);
    v(0, 0) = 3.0;
    v(1, 0) = -4.0;
    const auto n = norm(v, 2);
    ASSERT_TRUE(n.has_value());
    EXPECT_NEAR(*n, 5.0, 1e-12);
}

// ---- Schur decomposition preserves diagonal eigenvalues ----

TEST(LinalgEdgeTest, schur_diagonal_2x2) {
    // For a diagonal symmetric matrix, T should be upper triangular
    // with the same diagonal entries (eigenvalues)
    DMatrix A = zeros<double>(2, 2);
    A(0, 0) = 3.0;
    A(1, 1) = 7.0;
    const auto result = schur(A);
    ASSERT_TRUE(result.has_value());
    // T should be upper triangular; diagonal entries are eigenvalues
    EXPECT_NEAR(result->T(1, 0), 0.0, 1e-8);
    // Eigenvalues on diagonal in some order
    const double d0 = result->T(0, 0);
    const double d1 = result->T(1, 1);
    EXPECT_TRUE((std::abs(d0 - 3.0) < 1e-8 && std::abs(d1 - 7.0) < 1e-8) ||
                (std::abs(d0 - 7.0) < 1e-8 && std::abs(d1 - 3.0) < 1e-8));
    // Q must be square
    EXPECT_EQ(result->Q.rows(), 2u);
    EXPECT_EQ(result->Q.cols(), 2u);
}

// ---- diag construct and extract roundtrip ----

TEST(LinalgEdgeTest, diag_construct_and_extract) {
    const std::vector<double> vals{1.0, 5.0, 9.0};
    const DMatrix D = diag(vals);
    EXPECT_EQ(D.rows(), 3u);
    EXPECT_EQ(D.cols(), 3u);
    EXPECT_NEAR(D(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(D(1, 1), 5.0, 1e-12);
    EXPECT_NEAR(D(2, 2), 9.0, 1e-12);
    // Off-diagonal must be zero
    EXPECT_NEAR(D(0, 1), 0.0, 1e-12);
    EXPECT_NEAR(D(1, 2), 0.0, 1e-12);

    const auto extracted = diag(D);
    ASSERT_EQ(extracted.size(), 3u);
    EXPECT_NEAR(extracted[0], 1.0, 1e-12);
    EXPECT_NEAR(extracted[1], 5.0, 1e-12);
    EXPECT_NEAR(extracted[2], 9.0, 1e-12);
}

// ---- tril and triu on non-square matrix ----

TEST(LinalgEdgeTest, tril_triu_2x3) {
    DMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    const DMatrix L = tril(A);
    // Elements above diagonal (j > i) must be zero
    EXPECT_NEAR(L(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(L(0, 1), 0.0, 1e-12);
    EXPECT_NEAR(L(0, 2), 0.0, 1e-12);
    EXPECT_NEAR(L(1, 0), 4.0, 1e-12);
    EXPECT_NEAR(L(1, 1), 5.0, 1e-12);
    EXPECT_NEAR(L(1, 2), 0.0, 1e-12);

    const DMatrix U = triu(A);
    // Elements below diagonal (i > j) must be zero
    EXPECT_NEAR(U(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(U(0, 1), 2.0, 1e-12);
    EXPECT_NEAR(U(0, 2), 3.0, 1e-12);
    EXPECT_NEAR(U(1, 0), 0.0, 1e-12);
    EXPECT_NEAR(U(1, 1), 5.0, 1e-12);
    EXPECT_NEAR(U(1, 2), 6.0, 1e-12);
}
