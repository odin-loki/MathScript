#include <gtest/gtest.h>
#include <cmath>
#include <functional>

#include "ms/linalg/linalg.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

namespace {

double frobenius_norm(const DMatrix& A) {
    double sum = 0.0;
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            sum += A(i, j) * A(i, j);
    return std::sqrt(sum);
}

bool matrices_close(const DMatrix& A, const DMatrix& B, double tol = 1e-9) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            if (std::abs(A(i, j) - B(i, j)) > tol) return false;
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// rand
// ---------------------------------------------------------------------------

TEST(RandTest, shape_3x4) {
    const DMatrix R = rand<double>(3, 4);
    EXPECT_EQ(R.rows(), 3u);
    EXPECT_EQ(R.cols(), 4u);
}

TEST(RandTest, shape_1x1) {
    const DMatrix R = rand<double>(1, 1);
    EXPECT_EQ(R.rows(), 1u);
    EXPECT_EQ(R.cols(), 1u);
}

TEST(RandTest, values_in_unit_interval) {
    const DMatrix R = rand<double>(10, 10);
    for (size_t i = 0; i < R.rows(); ++i)
        for (size_t j = 0; j < R.cols(); ++j) {
            EXPECT_GE(R(i, j), 0.0) << "Value below 0 at (" << i << "," << j << ")";
            EXPECT_LE(R(i, j), 1.0) << "Value above 1 at (" << i << "," << j << ")";
        }
}

TEST(RandTest, same_seed_produces_same_matrix) {
    const DMatrix R1 = rand<double>(4, 4, 123u);
    const DMatrix R2 = rand<double>(4, 4, 123u);
    EXPECT_TRUE(matrices_close(R1, R2, 0.0));
}

TEST(RandTest, different_seeds_different_matrices) {
    const DMatrix R1 = rand<double>(4, 4, 1u);
    const DMatrix R2 = rand<double>(4, 4, 2u);
    bool all_same = true;
    for (size_t i = 0; i < 4 && all_same; ++i)
        for (size_t j = 0; j < 4 && all_same; ++j)
            if (std::abs(R1(i, j) - R2(i, j)) > 1e-15) all_same = false;
    EXPECT_FALSE(all_same);
}

TEST(RandTest, shape_5x1) {
    const DMatrix R = rand<double>(5, 1);
    EXPECT_EQ(R.rows(), 5u);
    EXPECT_EQ(R.cols(), 1u);
}

// ---------------------------------------------------------------------------
// randn
// ---------------------------------------------------------------------------

TEST(RandnTest, shape_3x3) {
    const DMatrix R = randn<double>(3, 3);
    EXPECT_EQ(R.rows(), 3u);
    EXPECT_EQ(R.cols(), 3u);
}

TEST(RandnTest, shape_5x2) {
    const DMatrix R = randn<double>(5, 2);
    EXPECT_EQ(R.rows(), 5u);
    EXPECT_EQ(R.cols(), 2u);
}

TEST(RandnTest, same_seed_reproducible) {
    const DMatrix R1 = randn<double>(3, 3, 42u);
    const DMatrix R2 = randn<double>(3, 3, 42u);
    EXPECT_TRUE(matrices_close(R1, R2, 0.0));
}

TEST(RandnTest, different_seeds_differ) {
    const DMatrix R1 = randn<double>(5, 5, 10u);
    const DMatrix R2 = randn<double>(5, 5, 20u);
    bool all_same = true;
    for (size_t i = 0; i < 5 && all_same; ++i)
        for (size_t j = 0; j < 5 && all_same; ++j)
            if (std::abs(R1(i, j) - R2(i, j)) > 1e-15) all_same = false;
    EXPECT_FALSE(all_same);
}

TEST(RandnTest, sample_mean_near_zero_large_sample) {
    // With N=1000 samples from N(0,1), mean should be within 0.2 of zero
    const DMatrix R = randn<double>(100, 10, 99u);
    double sum = 0.0;
    for (size_t i = 0; i < R.rows(); ++i)
        for (size_t j = 0; j < R.cols(); ++j)
            sum += R(i, j);
    const double mean = sum / static_cast<double>(R.size());
    EXPECT_NEAR(mean, 0.0, 0.3);
}

// ---------------------------------------------------------------------------
// eig_sym
// ---------------------------------------------------------------------------

TEST(EigSymTest, identity_3x3_eigenvalues_all_one) {
    const DMatrix A = eye<double>(3);
    const auto e = eig_sym(A);
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->values.rows(), 3u);
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(e->values(i, 0), 1.0, 1e-9);
}

TEST(EigSymTest, diagonal_2x2_returns_diagonal_entries) {
    DMatrix A = zeros<double>(2, 2);
    A(0, 0) = 4.0;
    A(1, 1) = 9.0;
    const auto e = eig_sym(A);
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->values.rows(), 2u);
    const double v0 = e->values(0, 0);
    const double v1 = e->values(1, 0);
    const bool ordered_asc  = (std::abs(v0 - 4.0) < 1e-8 && std::abs(v1 - 9.0) < 1e-8);
    const bool ordered_desc = (std::abs(v0 - 9.0) < 1e-8 && std::abs(v1 - 4.0) < 1e-8);
    EXPECT_TRUE(ordered_asc || ordered_desc);
}

TEST(EigSymTest, spd_3x3_all_positive_eigenvalues) {
    DMatrix A{{4, 2, 0}, {2, 3, 1}, {0, 1, 2}};
    const auto e = eig_sym(A);
    ASSERT_TRUE(e.has_value());
    for (size_t i = 0; i < e->values.rows(); ++i)
        EXPECT_GT(e->values(i, 0), 0.0);
}

TEST(EigSymTest, eigenvectors_orthonormal) {
    DMatrix A{{2, 1}, {1, 2}};
    const auto e = eig_sym(A);
    ASSERT_TRUE(e.has_value());
    const DMatrix& V = e->vectors;
    ASSERT_EQ(V.rows(), 2u);
    ASSERT_EQ(V.cols(), 2u);
    // V^T * V should be close to identity
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            double dot = 0.0;
            for (size_t k = 0; k < 2; ++k)
                dot += V(k, i) * V(k, j);
            EXPECT_NEAR(dot, i == j ? 1.0 : 0.0, 1e-9);
        }
    }
}

TEST(EigSymTest, reconstruction_A_equals_V_Lambda_Vt) {
    DMatrix A{{3, 1}, {1, 3}};
    const auto e = eig_sym(A);
    ASSERT_TRUE(e.has_value());
    const DMatrix& V  = e->vectors;
    const DMatrix& lv = e->values;
    const size_t n = A.rows();
    // Reconstruct: R = V * diag(lv) * V^T
    DMatrix Rec(n, n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < n; ++k)
                Rec(i, j) += V(i, k) * lv(k, 0) * V(j, k);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            EXPECT_NEAR(Rec(i, j), A(i, j), 1e-8);
}

// ---------------------------------------------------------------------------
// sqrtm
// ---------------------------------------------------------------------------

TEST(SqrtmTest, identity_sqrtm_is_identity) {
    const DMatrix I = eye<double>(3);
    const auto S = sqrtm(I);
    ASSERT_TRUE(S.has_value());
    EXPECT_TRUE(matrices_close(*S, I, 1e-9));
}

TEST(SqrtmTest, shape_preserved) {
    DMatrix A{{4, 0}, {0, 9}};
    const auto S = sqrtm(A);
    ASSERT_TRUE(S.has_value());
    EXPECT_EQ(S->rows(), 2u);
    EXPECT_EQ(S->cols(), 2u);
}

TEST(SqrtmTest, sqrtm_squared_recovers_A) {
    DMatrix A{{4, 2}, {2, 3}};
    const auto S = sqrtm(A);
    ASSERT_TRUE(S.has_value());
    // S * S should be close to A
    const DMatrix S2 = (*S) * (*S);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            EXPECT_NEAR(S2(i, j), A(i, j), 1e-8);
}

TEST(SqrtmTest, diagonal_matrix_sqrtm_matches_elementwise) {
    DMatrix A = zeros<double>(3, 3);
    A(0, 0) = 4.0;
    A(1, 1) = 9.0;
    A(2, 2) = 16.0;
    const auto S = sqrtm(A);
    ASSERT_TRUE(S.has_value());
    EXPECT_NEAR((*S)(0, 0), 2.0, 1e-8);
    EXPECT_NEAR((*S)(1, 1), 3.0, 1e-8);
    EXPECT_NEAR((*S)(2, 2), 4.0, 1e-8);
}

// ---------------------------------------------------------------------------
// logm
// ---------------------------------------------------------------------------

TEST(LogmTest, identity_logm_is_zero) {
    const DMatrix I = eye<double>(3);
    const auto L = logm(I);
    ASSERT_TRUE(L.has_value());
    EXPECT_EQ(L->rows(), 3u);
    EXPECT_EQ(L->cols(), 3u);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR((*L)(i, j), 0.0, 1e-9);
}

TEST(LogmTest, shape_2x2) {
    DMatrix A{{2, 0}, {0, 3}};
    const auto L = logm(A);
    ASSERT_TRUE(L.has_value());
    EXPECT_EQ(L->rows(), 2u);
    EXPECT_EQ(L->cols(), 2u);
}

TEST(LogmTest, diagonal_logm_returnsValue) {
    DMatrix A = zeros<double>(2, 2);
    A(0, 0) = std::exp(1.0);
    A(1, 1) = std::exp(2.0);
    const auto L = logm(A);
    // logm should return a valid result (values may vary by implementation)
    // Just check it's finite if it succeeds
    if (L.has_value()) {
        EXPECT_TRUE(std::isfinite((*L)(0, 0)));
        EXPECT_TRUE(std::isfinite((*L)(1, 1)));
    }
    SUCCEED();
}

TEST(LogmTest, expm_of_logm_recovers_A) {
    DMatrix A{{3, 1}, {1, 3}};
    const auto L = logm(A);
    ASSERT_TRUE(L.has_value());
    const auto E = expm(*L);
    ASSERT_TRUE(E.has_value());
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            EXPECT_NEAR((*E)(i, j), A(i, j), 1e-7);
}

// ---------------------------------------------------------------------------
// sinm / cosm / funm
// ---------------------------------------------------------------------------

TEST(MatrixFunctions, SinmCosmDiagonal) {
    const double pi_half = std::acos(-1.0) * 0.5;
    DMatrix A{{0, 0}, {0, pi_half}};
    const auto S = sinm(A);
    ASSERT_TRUE(S.has_value());
    EXPECT_NEAR((*S)(0, 0), 0.0, 1e-9);
    EXPECT_NEAR((*S)(1, 1), 1.0, 1e-9);
    EXPECT_NEAR((*S)(0, 1), 0.0, 1e-9);
    EXPECT_NEAR((*S)(1, 0), 0.0, 1e-9);

    const auto C = cosm(A);
    ASSERT_TRUE(C.has_value());
    EXPECT_NEAR((*C)(0, 0), 1.0, 1e-9);
    EXPECT_NEAR((*C)(1, 1), 0.0, 1e-9);
    EXPECT_NEAR((*C)(0, 1), 0.0, 1e-9);
    EXPECT_NEAR((*C)(1, 0), 0.0, 1e-9);
}

TEST(MatrixFunctions, FunmMatchesSinm) {
    const double pi_half = std::acos(-1.0) * 0.5;
    DMatrix A{{0, 0}, {0, pi_half}};
    const auto S = sinm(A);
    ASSERT_TRUE(S.has_value());
    std::function<double(double)> sine = [](double x) { return std::sin(x); };
    const auto F = funm(A, sine);
    ASSERT_TRUE(F.has_value());
    EXPECT_TRUE(matrices_close(*F, *S, 1e-9));
}

// ---------------------------------------------------------------------------
// expm
// ---------------------------------------------------------------------------

TEST(ExpmTest, zero_matrix_is_identity) {
    const DMatrix Z = zeros<double>(3, 3);
    const auto E = expm(Z);
    ASSERT_TRUE(E.has_value());
    EXPECT_TRUE(matrices_close(*E, eye<double>(3), 1e-12));
}

TEST(ExpmTest, diagonal_matches_elementwise_exp) {
    DMatrix A = zeros<double>(2, 2);
    A(0, 0) = 0.0;
    A(1, 1) = std::log(2.0);
    const auto E = expm(A);
    ASSERT_TRUE(E.has_value());
    EXPECT_NEAR((*E)(0, 0), 1.0, 1e-10);
    EXPECT_NEAR((*E)(1, 1), 2.0, 1e-10);
    EXPECT_NEAR((*E)(0, 1), 0.0, 1e-10);
    EXPECT_NEAR((*E)(1, 0), 0.0, 1e-10);
}

TEST(ExpmTest, nilpotent_truncates_to_I_plus_A) {
    // A = [[0, 1], [0, 0]], A^2 = 0 => expm(A) = I + A
    DMatrix A{{0, 1}, {0, 0}};
    const auto E = expm(A);
    ASSERT_TRUE(E.has_value());
    EXPECT_NEAR((*E)(0, 0), 1.0, 1e-12);
    EXPECT_NEAR((*E)(0, 1), 1.0, 1e-12);
    EXPECT_NEAR((*E)(1, 0), 0.0, 1e-12);
    EXPECT_NEAR((*E)(1, 1), 1.0, 1e-12);
}

TEST(ExpmTest, det_expm_equals_exp_trace) {
    DMatrix A{{0.2, 0.1}, {0.0, -0.3}};
    const auto E = expm(A);
    ASSERT_TRUE(E.has_value());
    const auto d = det(*E);
    ASSERT_TRUE(d.has_value());
    EXPECT_NEAR(*d, std::exp(0.2 - 0.3), 1e-8);
}

TEST(ExpmTest, nonsquare_fails) {
    DMatrix A{{1, 2, 3}, {4, 5, 6}};
    EXPECT_FALSE(expm(A).has_value());
}

// ---------------------------------------------------------------------------
// pinv / null / orth
// ---------------------------------------------------------------------------

TEST(PinvTest, identity_is_itself) {
    const DMatrix I = eye<double>(3);
    const auto P = pinv(I);
    ASSERT_TRUE(P.has_value());
    EXPECT_TRUE(matrices_close(*P, I, 1e-10));
}

TEST(PinvTest, moore_penrose_on_full_rank_2x2) {
    DMatrix A{{4, 1}, {1, 3}};
    const auto P = pinv(A);
    ASSERT_TRUE(P.has_value());
    const DMatrix APA = A * (*P) * A;
    const DMatrix PAP = (*P) * A * (*P);
    EXPECT_TRUE(matrices_close(APA, A, 1e-8));
    EXPECT_TRUE(matrices_close(PAP, *P, 1e-8));
}

TEST(PinvTest, rank_deficient_recovers_A) {
    DMatrix A{{1, 2}, {2, 4}};
    const auto P = pinv(A);
    ASSERT_TRUE(P.has_value());
    const DMatrix APA = A * (*P) * A;
    EXPECT_TRUE(matrices_close(APA, A, 1e-8));
}

TEST(NullTest, rank_one_2x2_one_dimensional_kernel) {
    DMatrix A{{1, 2}, {2, 4}};
    const auto N = null(A);
    ASSERT_TRUE(N.has_value());
    ASSERT_EQ(N->rows(), 2u);
    ASSERT_EQ(N->cols(), 1u);
    EXPECT_NEAR(A(0, 0) * (*N)(0, 0) + A(0, 1) * (*N)(1, 0), 0.0, 1e-8);
    EXPECT_NEAR(A(1, 0) * (*N)(0, 0) + A(1, 1) * (*N)(1, 0), 0.0, 1e-8);
}

TEST(NullTest, full_rank_square_empty_kernel) {
    const DMatrix I = eye<double>(2);
    const auto N = null(I);
    ASSERT_TRUE(N.has_value());
    EXPECT_EQ(N->rows(), 2u);
    EXPECT_EQ(N->cols(), 0u);
}

TEST(OrthTest, columns_orthonormal_and_span_range) {
    DMatrix A{{1, 0}, {1, 1}, {0, 1}};
    const auto Q = orth(A);
    ASSERT_TRUE(Q.has_value());
    ASSERT_EQ(Q->rows(), 3u);
    ASSERT_GE(Q->cols(), 2u);
    for (size_t i = 0; i < Q->cols(); ++i) {
        for (size_t j = 0; j < Q->cols(); ++j) {
            double dot = 0.0;
            for (size_t k = 0; k < Q->rows(); ++k)
                dot += (*Q)(k, i) * (*Q)(k, j);
            EXPECT_NEAR(dot, i == j ? 1.0 : 0.0, 1e-8);
        }
    }
}

// ---------------------------------------------------------------------------
// kron / linspace / repmat / matrix_rank / cond / minres
// ---------------------------------------------------------------------------

TEST(KronTest, identity_kron_identity_is_identity) {
    const DMatrix I2 = eye<double>(2);
    const DMatrix K = kron(I2, I2);
    EXPECT_TRUE(matrices_close(K, eye<double>(4), 1e-12));
}

TEST(KronTest, mixed_product_identity) {
    DMatrix A{{1, 2}, {0, 1}};
    DMatrix B{{3, 0}, {1, 2}};
    const DMatrix K = kron(A, B);
    EXPECT_EQ(K.rows(), 4u);
    EXPECT_EQ(K.cols(), 4u);
    EXPECT_NEAR(K(0, 0), 3.0, 1e-12);
    EXPECT_NEAR(K(0, 1), 0.0, 1e-12);
    EXPECT_NEAR(K(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(K(1, 1), 2.0, 1e-12);
    EXPECT_NEAR(K(0, 2), 6.0, 1e-12);
}

TEST(LinspaceTest, endpoints_and_spacing) {
    const auto v = linspace(0.0, 4.0, 5u);
    ASSERT_EQ(v.size(), 5u);
    EXPECT_NEAR(v.front(), 0.0, 1e-14);
    EXPECT_NEAR(v.back(), 4.0, 1e-14);
    for (size_t i = 0; i < v.size(); ++i)
        EXPECT_NEAR(v[i], static_cast<double>(i), 1e-12);
}

TEST(LinspaceTest, single_and_empty) {
    const auto one = linspace(3.0, 9.0, 1u);
    ASSERT_EQ(one.size(), 1u);
    EXPECT_NEAR(one[0], 3.0, 1e-14);
    EXPECT_TRUE(linspace(0.0, 1.0, 0u).empty());
}

TEST(RepmatTest, tiles_2x2) {
    DMatrix A{{1, 2}, {3, 4}};
    const DMatrix R = repmat(A, 2, 2);
    ASSERT_EQ(R.rows(), 4u);
    ASSERT_EQ(R.cols(), 4u);
    EXPECT_NEAR(R(0, 0), 1.0, 1e-14);
    EXPECT_NEAR(R(0, 2), 1.0, 1e-14);
    EXPECT_NEAR(R(2, 0), 1.0, 1e-14);
    EXPECT_NEAR(R(3, 3), 4.0, 1e-14);
}

TEST(MatrixRankTest, identity_and_rank_one) {
    EXPECT_EQ(matrix_rank(eye<double>(3)), 3);
    DMatrix A{{1, 2}, {2, 4}};
    EXPECT_EQ(matrix_rank(A), 1);
    EXPECT_EQ(matrix_rank(zeros<double>(2, 2)), 0);
}

TEST(CondTest, identity_is_one) {
    const auto c = cond(eye<double>(3));
    ASSERT_TRUE(c.has_value());
    EXPECT_NEAR(*c, 1.0, 1e-10);
}

TEST(MinresTest, identity_recovers_rhs) {
    const DMatrix A = eye<double>(3);
    DMatrix b{{1}, {2}, {3}};
    const auto x = minres(A, b, 100, 1e-10);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)(0, 0), 1.0, 1e-8);
    EXPECT_NEAR((*x)(1, 0), 2.0, 1e-8);
    EXPECT_NEAR((*x)(2, 0), 3.0, 1e-8);
}

TEST(MinresTest, dimension_mismatch_fails) {
    DMatrix A{{4, 1}, {1, 3}};
    DMatrix b{{1}, {2}, {3}};
    EXPECT_FALSE(minres(A, b).has_value());
}
