#include <cmath>
#include <vector>
#include <gtest/gtest.h>

#include "ms/error/error_types.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

namespace {

// Reconstructs U * diag(S) * V^T and asserts it matches A column-by-column
// (not just via an aggregate norm), since the regression covered here
// specifically corrupted later columns of the SVD reconstruction while
// leaving earlier ones correct.
void expect_svd_reconstructs(const DMatrix& A, double tol = 1e-9) {
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    DMatrix Sigma = zeros<double>(result->S.rows(), result->S.rows());
    for (size_t i = 0; i < result->S.rows(); ++i) {
        Sigma(i, i) = result->S(i, 0);
    }
    const DMatrix reconstructed = result->U * Sigma * transpose(result->V);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(reconstructed(i, j), A(i, j), tol) << "mismatch at (i=" << i << ", j=" << j << ")";
        }
    }
}

// ||M^T M - I||_F for an m x n column-orthonormal M (used to check the
// significant, well-separated singular vectors of U/V remain orthonormal;
// vectors tied to negligible/degenerate singular values are intentionally
// excluded since their basis is mathematically arbitrary).
double ortho_error(const DMatrix& M, size_t ncols) {
    const size_t m = M.rows();
    double err = 0.0;
    for (size_t i = 0; i < ncols; ++i) {
        for (size_t j = 0; j < ncols; ++j) {
            double dot = 0.0;
            for (size_t k = 0; k < m; ++k) {
                dot += M(k, i) * M(k, j);
            }
            const double expected = (i == j) ? 1.0 : 0.0;
            err += (dot - expected) * (dot - expected);
        }
    }
    return std::sqrt(err);
}

} // namespace

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

// ---------------------------------------------------------------------------
// Regression tests for the dorgbr('P', ...) P/P**T swap bug: for any m>=n
// bidiagonalized matrix whose P-transform needs 2+ Householder reflectors,
// dorgbr previously returned P instead of the documented P**T, corrupting
// dgesvd_tall's back-transformed right singular vectors for every column
// beyond the ones spanned by the first reflector. This was most easily
// exposed by rank-deficient matrices (several trailing near-zero singular
// values beyond the leading, well-separated one(s)), which is how it was
// originally found, but the underlying defect affected full-rank inputs too.
// ---------------------------------------------------------------------------

TEST(LinalgDecompTest, svd_rank2_4x4_exact_zero_svs_reconstructs) {
    // A = 5*u1*v1^T + 3*u2*v2^T with u1=(1,1,1,1)/2, u2=(1,-1,1,-1)/2,
    // v1=(1,1,0,0)/sqrt2, v2=(0,0,1,1)/sqrt2 all orthonormal, so A is exactly
    // rank 2 (singular values 5, 3, 0, 0). Before the fix, columns 2 and 3 of
    // U*S*V^T did not match A while columns 0 and 1 did.
    const double u1[4] = {0.5, 0.5, 0.5, 0.5};
    const double u2[4] = {0.5, -0.5, 0.5, -0.5};
    const double v1[4] = {1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0), 0.0, 0.0};
    const double v2[4] = {0.0, 0.0, 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0)};
    DMatrix A(4, 4, 0.0);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            A(i, j) = 5.0 * u1[i] * v1[j] + 3.0 * u2[i] * v2[j];
        }
    }

    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->S(0, 0), 5.0, 1e-9);
    EXPECT_NEAR(result->S(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(result->S(2, 0), 0.0, 1e-9);
    EXPECT_NEAR(result->S(3, 0), 0.0, 1e-9);
    expect_svd_reconstructs(A);
    EXPECT_LT(ortho_error(result->U, 2), 1e-8);
    EXPECT_LT(ortho_error(result->V, 2), 1e-8);
}

TEST(LinalgDecompTest, svd_rank1_5x5_many_trailing_near_zero_svs_reconstructs) {
    // Rank-1 5x5 matrix: 4 of the 5 singular values are numerically zero, well
    // beyond the leading one, stressing the same reflector back-transform bug
    // with more degenerate columns than the 4x4 case above.
    std::vector<double> u1 = {0.2, 0.4, 0.4, 0.6, 0.5196152423};
    std::vector<double> v1 = {0.5, -0.3, 0.4, -0.5, 0.5};
    auto normalize = [](std::vector<double>& v) {
        double s = 0.0;
        for (double x : v) s += x * x;
        s = std::sqrt(s);
        for (double& x : v) x /= s;
    };
    normalize(u1);
    normalize(v1);
    DMatrix A(5, 5, 0.0);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            A(i, j) = 9.0 * u1[i] * v1[j];
        }
    }

    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->S(0, 0), 9.0, 1e-9);
    for (size_t i = 1; i < 5; ++i) {
        EXPECT_NEAR(result->S(i, 0), 0.0, 1e-8);
    }
    expect_svd_reconstructs(A, 1e-7);
    EXPECT_LT(ortho_error(result->U, 1), 1e-8);
    EXPECT_LT(ortho_error(result->V, 1), 1e-8);
}

TEST(LinalgDecompTest, svd_rank_deficient_wide_3x5_reconstructs) {
    // Wide (m < n) rank-2 matrix: exercises dgesvd_wide, which internally
    // transposes and calls dgesvd_tall, so it shares the same P/P**T path.
    const double u1[3] = {0.6, 0.8, 0.0};
    const double u2[3] = {-0.8, 0.6, 0.0};
    std::vector<double> v1 = {0.2, 0.4, 0.4, 0.6, 0.5196152423};
    std::vector<double> v2 = {0.5, -0.3, 0.4, -0.5, 0.5};
    auto normalize = [](std::vector<double>& v) {
        double s = 0.0;
        for (double x : v) s += x * x;
        s = std::sqrt(s);
        for (double& x : v) x /= s;
    };
    normalize(v1);
    double dot = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) dot += v1[i] * v2[i];
    for (size_t i = 0; i < v1.size(); ++i) v2[i] -= dot * v1[i];
    normalize(v2);

    DMatrix A(3, 5, 0.0);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            A(i, j) = 7.0 * u1[i] * v1[j] + 2.0 * u2[i] * v2[j];
        }
    }

    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->S(0, 0), 7.0, 1e-9);
    EXPECT_NEAR(result->S(1, 0), 2.0, 1e-9);
    EXPECT_NEAR(result->S(2, 0), 0.0, 1e-9);
    expect_svd_reconstructs(A);
}

TEST(LinalgDecompTest, svd_rank_deficient_tall_5x3_reconstructs) {
    // Tall (m > n) counterpart of the wide case above, rank 2 out of 3.
    std::vector<double> u1 = {0.2, 0.4, 0.4, 0.6, 0.5196152423};
    std::vector<double> u2 = {0.5, -0.3, 0.4, -0.5, 0.5};
    auto normalize = [](std::vector<double>& v) {
        double s = 0.0;
        for (double x : v) s += x * x;
        s = std::sqrt(s);
        for (double& x : v) x /= s;
    };
    normalize(u1);
    double dot = 0.0;
    for (size_t i = 0; i < u1.size(); ++i) dot += u1[i] * u2[i];
    for (size_t i = 0; i < u1.size(); ++i) u2[i] -= dot * u1[i];
    normalize(u2);
    const double v1[3] = {0.6, 0.8, 0.0};
    const double v2[3] = {-0.8, 0.6, 0.0};

    DMatrix A(5, 3, 0.0);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            A(i, j) = 7.0 * u1[i] * v1[j] + 2.0 * u2[i] * v2[j];
        }
    }

    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->S(0, 0), 7.0, 1e-9);
    EXPECT_NEAR(result->S(1, 0), 2.0, 1e-9);
    EXPECT_NEAR(result->S(2, 0), 0.0, 1e-9);
    expect_svd_reconstructs(A);
}

TEST(LinalgDecompTest, svd_rank_deficient_square_6x6_half_rank_reconstructs) {
    // 6x6 with true rank exactly half the dimension (3 of 6 singular values
    // zero), covering a "rank = half of min(m,n)" configuration distinct
    // from the rank-1/rank-2 cases above.
    DMatrix U0{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    DMatrix V0{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
    // Orthonormalize the columns of U0 and V0 (Gram-Schmidt) so the outer-
    // product sum below has exact singular values 6, 4, 2.
    auto gram_schmidt = [](DMatrix& M) {
        for (size_t j = 0; j < M.cols(); ++j) {
            for (size_t p = 0; p < j; ++p) {
                double dot = 0.0;
                for (size_t i = 0; i < M.rows(); ++i) dot += M(i, j) * M(i, p);
                for (size_t i = 0; i < M.rows(); ++i) M(i, j) -= dot * M(i, p);
            }
            double norm = 0.0;
            for (size_t i = 0; i < M.rows(); ++i) norm += M(i, j) * M(i, j);
            norm = std::sqrt(norm);
            for (size_t i = 0; i < M.rows(); ++i) M(i, j) /= norm;
        }
    };
    gram_schmidt(U0);
    gram_schmidt(V0);

    const double sigmas[3] = {6.0, 4.0, 2.0};
    DMatrix A(6, 6, 0.0);
    for (size_t k = 0; k < 3; ++k) {
        for (size_t i = 0; i < 6; ++i) {
            for (size_t j = 0; j < 6; ++j) {
                A(i, j) += sigmas[k] * U0(i, k) * V0(j, k);
            }
        }
    }

    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->S(0, 0), 6.0, 1e-8);
    EXPECT_NEAR(result->S(1, 0), 4.0, 1e-8);
    EXPECT_NEAR(result->S(2, 0), 2.0, 1e-8);
    for (size_t i = 3; i < 6; ++i) {
        EXPECT_NEAR(result->S(i, 0), 0.0, 1e-7);
    }
    expect_svd_reconstructs(A, 1e-7);
    EXPECT_LT(ortho_error(result->U, 3), 1e-7);
    EXPECT_LT(ortho_error(result->V, 3), 1e-7);
}

TEST(LinalgDecompTest, svd_fullrank_5x5_reconstructs) {
    // Full-rank (no rank deficiency at all) 5x5 matrix. The P/P**T bug fixed
    // here was not limited to rank-deficient inputs — it silently corrupted
    // reconstruction for full-rank m>=n matrices with n>=3 too, so this case
    // guards against regressing the common, non-degenerate path.
    DMatrix A{{4, 1, 0, 0, 2}, {1, 3, 1, 0, 0}, {0, 1, 5, 1, 0}, {0, 0, 1, 6, 1}, {2, 0, 0, 1, 4}};
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i + 1 < result->S.rows(); ++i) {
        EXPECT_GE(result->S(i, 0), result->S(i + 1, 0));
        EXPECT_GT(result->S(i, 0), 1e-6);
    }
    expect_svd_reconstructs(A);
    EXPECT_LT(ortho_error(result->U, 5), 1e-8);
    EXPECT_LT(ortho_error(result->V, 5), 1e-8);
}

TEST(LinalgDecompTest, svd_fullrank_tall_6x4_reconstructs) {
    // Full-rank tall regression companion to the rank-deficient tall case.
    DMatrix A{{4, 1, 0, 2}, {1, 3, 1, 0}, {0, 1, 5, 1}, {2, 0, 1, 4}, {1, 2, 0, 1}, {0, 1, 2, 3}};
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->S.rows(), 4u);
    for (size_t i = 0; i < result->S.rows(); ++i) {
        EXPECT_GT(result->S(i, 0), 1e-6);
    }
    expect_svd_reconstructs(A);
    EXPECT_LT(ortho_error(result->V, 4), 1e-8);
}

TEST(LinalgDecompTest, svd_rank_deficient_singular_values_sorted_nonnegative) {
    // Sanity check that singular values stay sorted descending and
    // non-negative across the rank-deficient reconstructions above.
    std::vector<double> u1 = {0.2, 0.4, 0.4, 0.6, 0.5196152423};
    std::vector<double> v1 = {0.5, -0.3, 0.4, -0.5, 0.5};
    auto normalize = [](std::vector<double>& v) {
        double s = 0.0;
        for (double x : v) s += x * x;
        s = std::sqrt(s);
        for (double& x : v) x /= s;
    };
    normalize(u1);
    normalize(v1);
    DMatrix A(5, 5, 0.0);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            A(i, j) = 9.0 * u1[i] * v1[j];
        }
    }
    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < result->S.rows(); ++i) {
        EXPECT_GE(result->S(i, 0), 0.0);
    }
    for (size_t i = 0; i + 1 < result->S.rows(); ++i) {
        EXPECT_GE(result->S(i, 0), result->S(i + 1, 0) - 1e-9);
    }
}

TEST(LinalgDecompTest, svd_negative_bidiagonal_entries_orthogonal_vectors) {
    // Regression test for the actual root cause: dbdsqr's Givens-rotation
    // generator (dlartg) computed r = hypot(...) with a sign that didn't
    // match cs*f + sn*g whenever the larger-magnitude input was negative.
    // This produced a self-consistent (reconstructs fine) but wrong,
    // non-orthogonal U/V whenever the bidiagonal reduction of A produced a
    // diagonal/off-diagonal entry with that sign pattern partway through the
    // Givens sweep - which happens for ordinary (not just rank-deficient)
    // matrices, since dgebrd-produced bidiagonal entries can be of either
    // sign. A 5x5 matrix already bidiagonal with one negative interior
    // diagonal entry reproduces it directly at the dbdsqr level.
    DMatrix A(5, 5, 0.0);
    const double d[5] = {2, 3, -4, 5, 6};
    const double e[4] = {1, 1.3, 1.6, 1.9};
    for (size_t i = 0; i < 5; ++i) A(i, i) = d[i];
    for (size_t i = 0; i < 4; ++i) A(i, i + 1) = e[i];

    auto result = svd(A);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i + 1 < result->S.rows(); ++i) {
        EXPECT_GE(result->S(i, 0), result->S(i + 1, 0));
    }
    for (size_t i = 0; i < result->S.rows(); ++i) {
        EXPECT_GE(result->S(i, 0), 0.0);
    }
    expect_svd_reconstructs(A, 1e-9);
    EXPECT_LT(ortho_error(result->U, 5), 1e-9);
    EXPECT_LT(ortho_error(result->V, 5), 1e-9);
}

TEST(LinalgDecompTest, svd_negative_bidiagonal_entries_n3_and_n4) {
    // Smaller-size companions of the case above (the bug required at least
    // n=3, since n=2 is solved directly via dlasv2 without a Givens sweep).
    {
        DMatrix A(3, 3, 0.0);
        const double d[3] = {2, -3, 4};
        const double e[2] = {1, 1.3};
        for (size_t i = 0; i < 3; ++i) A(i, i) = d[i];
        for (size_t i = 0; i < 2; ++i) A(i, i + 1) = e[i];
        auto result = svd(A);
        ASSERT_TRUE(result.has_value());
        expect_svd_reconstructs(A, 1e-9);
        EXPECT_LT(ortho_error(result->U, 3), 1e-9);
        EXPECT_LT(ortho_error(result->V, 3), 1e-9);
    }
    {
        DMatrix A(4, 4, 0.0);
        const double d[4] = {2, 3, -4, 5};
        const double e[3] = {1, 1.3, 1.6};
        for (size_t i = 0; i < 4; ++i) A(i, i) = d[i];
        for (size_t i = 0; i < 3; ++i) A(i, i + 1) = e[i];
        auto result = svd(A);
        ASSERT_TRUE(result.has_value());
        expect_svd_reconstructs(A, 1e-9);
        EXPECT_LT(ortho_error(result->U, 4), 1e-9);
        EXPECT_LT(ortho_error(result->V, 4), 1e-9);
    }
}

TEST(LinalgDecompTest, svd_rank_deficient_null_and_orth_regression) {
    // null() (via eig_sym on the Gram matrix) and orth() (via qr()) don't
    // actually call svd() in this codebase, but they're part of the same
    // rank-revealing family the user asked to regression-check alongside
    // pinv() (which does call svd()) against a rank-deficient input.
    std::vector<double> u1 = {0.2, 0.4, 0.4, 0.6, 0.5196152423};
    std::vector<double> v1 = {0.5, -0.3, 0.4, -0.5, 0.5};
    auto normalize = [](std::vector<double>& v) {
        double s = 0.0;
        for (double x : v) s += x * x;
        s = std::sqrt(s);
        for (double& x : v) x /= s;
    };
    normalize(u1);
    normalize(v1);
    DMatrix A(5, 5, 0.0);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            A(i, j) = 9.0 * u1[i] * v1[j];
        }
    }

    auto N = null(A);
    ASSERT_TRUE(N.has_value());
    EXPECT_GE(N->cols(), 3u);
    const DMatrix AN = A * N.value();
    for (size_t i = 0; i < AN.rows(); ++i) {
        for (size_t j = 0; j < AN.cols(); ++j) {
            EXPECT_NEAR(AN(i, j), 0.0, 1e-6);
        }
    }

    auto O = orth(A);
    ASSERT_TRUE(O.has_value());
    EXPECT_EQ(O->cols(), 1u);
    DMatrix OtO = transpose(O.value()) * O.value();
    EXPECT_NEAR(OtO(0, 0), 1.0, 1e-6);
}

TEST(LinalgDecompTest, svd_rank_deficient_pinv_regression) {
    // pinv() is built directly on svd(); confirm the Moore-Penrose identity
    // A * pinv(A) * A == A still holds for a rank-deficient input after the fix.
    std::vector<double> u1 = {0.2, 0.4, 0.4, 0.6, 0.5196152423};
    std::vector<double> v1 = {0.5, -0.3, 0.4, -0.5, 0.5};
    auto normalize = [](std::vector<double>& v) {
        double s = 0.0;
        for (double x : v) s += x * x;
        s = std::sqrt(s);
        for (double& x : v) x /= s;
    };
    normalize(u1);
    normalize(v1);
    DMatrix A(5, 5, 0.0);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            A(i, j) = 9.0 * u1[i] * v1[j];
        }
    }
    auto Ap = pinv(A);
    ASSERT_TRUE(Ap.has_value());
    const DMatrix AApA = A * Ap.value() * A;
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(AApA(i, j), A(i, j), 1e-6);
        }
    }
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
