#include <cmath>
#include <gtest/gtest.h>

#include "ms/error/error_types.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

TEST(LinalgDecompTest, nonsymmetric_eig) {
    DMatrix A{{0, 1}, {-1, 0}};
    auto result = eig(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->values.rows(), 2u);
}

TEST(LinalgDecompTest, schur_factorization) {
    DMatrix A{{1, 2, 3}, {0, 4, 5}, {0, 0, 6}};
    auto result = schur(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->T.rows(), 3u);
    EXPECT_EQ(result->Q.cols(), 3u);
}

TEST(LinalgDecompTest, bidiagonal_reduction) {
    DMatrix A{{1, 2}, {3, 4}, {5, 6}};
    auto result = bidiag(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->B.rows(), 3u);
    EXPECT_EQ(result->B.cols(), 2u);
}

TEST(LinalgDecompTest, hessenberg_form) {
    DMatrix A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto H = hess(A);
    ASSERT_TRUE(H.has_value());
    EXPECT_EQ(H->rows(), 3u);
    EXPECT_NEAR((*H)(2, 0), 0.0, 1e-10);
}

TEST(LinalgDecompTest, logm_diagonal) {
    DMatrix A = eye<double>(2);
    A(0, 0) = 2.0;
    A(1, 1) = 3.0;
    auto L = logm(A);
    ASSERT_TRUE(L.has_value());
    EXPECT_NEAR((*L)(0, 0), std::log(2.0), 1e-6);
    EXPECT_NEAR((*L)(1, 1), std::log(3.0), 1e-6);
}

TEST(LinalgDecompTest, wide_svd) {
    DMatrix A{{1, 2, 3}, {4, 5, 6}};
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->S.rows(), 2u);
    EXPECT_GT(result->S(0, 0), 0.0);
}

TEST(LinalgDecompTest, lsq_underdetermined) {
    DMatrix A{{1, 2, 3}};
    DMatrix b{{6}};
    auto x = lsq(A, b);
    ASSERT_TRUE(x.has_value());
    EXPECT_EQ(x->cols(), 1u);
    EXPECT_EQ(x->rows(), 3u);
}

TEST(LinalgDecompTest, eig_and_schur_4x4) {
    DMatrix A{{4, 1, 0, 0}, {1, 3, 1, 0}, {0, 1, 2, 1}, {0, 0, 1, 5}};
    auto eig_result = eig(A);
    ASSERT_TRUE(eig_result.has_value());
    EXPECT_EQ(eig_result->values.rows(), 4u);
    auto schur_result = schur(A);
    ASSERT_TRUE(schur_result.has_value());
    EXPECT_EQ(schur_result->T.rows(), 4u);
}

TEST(LinalgDecompTest, hess_and_schur_5x5) {
    DMatrix A{
        {1, 2, 0, 0, 0},
        {3, 4, 1, 0, 0},
        {0, 5, 6, 1, 0},
        {0, 0, 7, 8, 1},
        {0, 0, 0, 9, 10}};
    auto H = hess(A);
    ASSERT_TRUE(H.has_value());
    EXPECT_NEAR((*H)(2, 0), 0.0, 1e-9);
    EXPECT_NEAR((*H)(3, 0), 0.0, 1e-9);
    auto S = schur(A);
    ASSERT_TRUE(S.has_value());
    EXPECT_EQ(S->T.rows(), 5u);
}

TEST(LinalgDecompTest, ldl_3x3) {
    DMatrix A{{4, 1, 0}, {1, 3, 1}, {0, 1, 2}};
    auto result = ldl(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->L.rows(), 3u);
}

TEST(LinalgDecompTest, eig_dimension_and_domain_errors) {
    DMatrix rectangular{{1, 2, 3}, {4, 5, 6}};
    EXPECT_FALSE(eig(rectangular).has_value());
    EXPECT_FALSE(eig_sym(rectangular).has_value());

    DMatrix nonsymmetric{{1, 2}, {3, 4}};
    EXPECT_FALSE(eig_sym(nonsymmetric).has_value());
    EXPECT_NE(format_error(eig_sym(nonsymmetric).error()).find("symmetric"), std::string::npos);
}

TEST(LinalgDecompTest, eig_nonsymmetric_2x2_known_values) {
    DMatrix A{{3, 1}, {0, 1}};
    auto result = eig(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->values(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(result->values(1, 0), 1.0, 1e-9);
}

TEST(LinalgDecompTest, svd_empty_matrix_error) {
    DMatrix empty(0, 2);
    EXPECT_FALSE(svd(empty).has_value());
}

TEST(LinalgDecompTest, svd_rank_deficient_reconstructs) {
    DMatrix A{{1, 2, 3}, {2, 4, 6}, {1, 1, 1}};
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->S(2, 0), 0.0, 1e-10);
    DMatrix Sigma = zeros<double>(result->S.rows(), result->S.rows());
    for (size_t i = 0; i < result->S.rows(); ++i) {
        Sigma(i, i) = result->S(i, 0);
    }
    const DMatrix reconstructed = result->U * Sigma * transpose(result->V);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(reconstructed(i, j), A(i, j), 1e-6);
        }
    }
}

TEST(LinalgDecompTest, svd_zero_cols_error) {
    DMatrix empty(2, 0);
    EXPECT_FALSE(svd(empty).has_value());
}

TEST(LinalgDecompTest, svd_tall_col_major_reconstructs) {
    DMatrix A{{1, 0}, {0, 2}, {1, 1}};
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->S.rows(), 2u);
    DMatrix Sigma = zeros<double>(result->S.rows(), result->S.rows());
    for (size_t i = 0; i < result->S.rows(); ++i) {
        Sigma(i, i) = result->S(i, 0);
    }
    const DMatrix reconstructed = result->U * Sigma * transpose(result->V);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(reconstructed(i, j), A(i, j), 1e-6);
        }
    }
}

TEST(LinalgDecompTest, svd_wide_col_major_reconstructs) {
    DMatrix A{{1, 2, 3, 4}, {5, 6, 7, 8}};
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->S.rows(), 2u);
    DMatrix Sigma = zeros<double>(result->S.rows(), result->S.rows());
    for (size_t i = 0; i < result->S.rows(); ++i) {
        Sigma(i, i) = result->S(i, 0);
    }
    const DMatrix reconstructed = result->U * Sigma * transpose(result->V);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(reconstructed(i, j), A(i, j), 1e-6);
        }
    }
}

TEST(LinalgDecompTest, svd_square_col_major_singular_values) {
    DMatrix A{{3, 1}, {1, 2}};
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->S(0, 0), 3.618033988749895, 1e-9);
    EXPECT_NEAR(result->S(1, 0), 1.381966011250105, 1e-9);
}

TEST(LinalgDecompTest, svd_row_major_wide_rank_deficient) {
    RowMatrix<double> A{{1, 2, 3, 4}, {2, 4, 6, 8}};
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->S(1, 0), 0.0, 1e-10);
    EXPECT_GT(result->S(0, 0), 0.0);
}

TEST(LinalgDecompTest, SchurTest_ReconstructionQTQt) {
    DMatrix A{{4, 3}, {6, 3}};
    const auto decomp = schur(A);
    ASSERT_TRUE(decomp.has_value());
    const DMatrix reconstructed = decomp->Q * decomp->T * transpose(decomp->Q);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(reconstructed(i, j), A(i, j), 1e-6);
        }
    }
}

TEST(LinalgDecompTest, SqrtmTest_SPD_Roundtrip) {
    DMatrix A{{4, 1}, {1, 3}};
    const auto S = sqrtm(A);
    ASSERT_TRUE(S.has_value());
    const DMatrix roundtrip = (*S) * (*S);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(roundtrip(i, j), A(i, j), 1e-9);
        }
    }
}

TEST(LinalgDecompTest, LogmTest_NonDiagonal_2x2) {
    DMatrix B{{0.05, 0.02}, {0.02, -0.03}};
    const auto A = expm(B);
    ASSERT_TRUE(A.has_value());
    const auto L = logm(*A);
    ASSERT_TRUE(L.has_value());
    for (size_t i = 0; i < B.rows(); ++i) {
        for (size_t j = 0; j < B.cols(); ++j) {
            EXPECT_NEAR((*L)(i, j), B(i, j), 1e-4);
        }
    }
}

TEST(LinalgDecompTest, SchurTest_EigenvaluesOnDiagonal) {
    DMatrix A{{4, 3}, {6, 3}};
    const auto decomp = schur(A);
    ASSERT_TRUE(decomp.has_value());
    const auto evals = eig(A);
    ASSERT_TRUE(evals.has_value());
    const double t00 = decomp->T(0, 0);
    const double t11 = decomp->T(1, 1);
    const double e0 = evals->values(0, 0);
    const double e1 = evals->values(1, 0);
    EXPECT_NEAR(t00, e0, 1e-4);
    EXPECT_NEAR(t11, e1, 1e-4);
}
