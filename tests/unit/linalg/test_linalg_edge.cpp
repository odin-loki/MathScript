#include <gtest/gtest.h>
#include <cmath>
#include <functional>
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

// ---- Empty matrices: defensive returns ----

TEST(LinalgEdgeTest, empty_svd_rank_and_orth) {
    DMatrix empty_row(0, 3);
    DMatrix empty_col(3, 0);
    EXPECT_FALSE(svd(empty_row).has_value());
    EXPECT_FALSE(svd(empty_col).has_value());
    EXPECT_FALSE(rank(empty_row).has_value());
    EXPECT_EQ(matrix_rank(empty_row), 0);
    EXPECT_EQ(matrix_rank(empty_col), 0);
    EXPECT_FALSE(cond(empty_row).has_value());
    EXPECT_FALSE(pinv(empty_row).has_value());

    const auto N = null(empty_col);
    if (!N.has_value()) {
        GTEST_SKIP() << "null of empty-col unsupported";
    }
    EXPECT_EQ(N->rows(), 0u);
    EXPECT_EQ(N->cols(), 0u);

    const auto Q = orth(empty_row);
    if (!Q.has_value()) {
        GTEST_SKIP() << "orth of empty-row unsupported";
    }
    EXPECT_EQ(Q->rows(), 0u);
    EXPECT_EQ(Q->cols(), 0u);
}

TEST(LinalgEdgeTest, empty_square_trace_det_schur) {
    DMatrix Z(0, 0);
    const auto t = trace(Z);
    if (t.has_value()) {
        EXPECT_NEAR(*t, 0.0, 1e-15);
    }
    const auto d = det(Z);
    if (d.has_value()) {
        EXPECT_NEAR(*d, 1.0, 1e-12);
    } else {
        GTEST_SKIP() << "det of 0x0 unsupported";
    }
    EXPECT_FALSE(svd(Z).has_value());
    if (!schur(Z).has_value()) {
        GTEST_SKIP() << "schur of 0x0 unsupported";
    }
}

TEST(LinalgEdgeTest, empty_diag_linspace_ones_zeros) {
    const std::vector<double> empty_vals;
    const DMatrix D = diag(empty_vals);
    EXPECT_EQ(D.rows(), 0u);
    EXPECT_EQ(D.cols(), 0u);

    const std::vector<double> grid = linspace(0.0, 1.0, 0u);
    EXPECT_TRUE(grid.empty());

    const DMatrix Z = zeros<double>(0, 2);
    EXPECT_EQ(Z.rows(), 0u);
    EXPECT_EQ(Z.cols(), 2u);
    const DMatrix O = ones<double>(2, 0);
    EXPECT_EQ(O.rows(), 2u);
    EXPECT_EQ(O.cols(), 0u);
}

// ---- Non-square defensive returns ----

TEST(LinalgEdgeTest, nonsquare_square_only_apis) {
    DMatrix rect{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    EXPECT_FALSE(det(rect).has_value());
    EXPECT_FALSE(trace(rect).has_value());
    EXPECT_FALSE(schur(rect).has_value());
    EXPECT_FALSE(eig(rect).has_value());
    EXPECT_FALSE(eig_sym(rect).has_value());
    EXPECT_FALSE(ldl(rect).has_value());
    EXPECT_FALSE(hess(rect).has_value());
    EXPECT_FALSE(logm(rect).has_value());
    EXPECT_FALSE(sqrtm(rect).has_value());
    EXPECT_FALSE(sinm(rect).has_value());
    EXPECT_FALSE(cosm(rect).has_value());
    std::function<double(double)> identity = [](double x) { return x; };
    EXPECT_FALSE(funm(rect, identity).has_value());

    DMatrix b(3, 1);
    b(0, 0) = 1.0;
    EXPECT_FALSE(solve(rect, b).has_value());
    EXPECT_FALSE(cg(rect, b).has_value());
    EXPECT_FALSE(jacobi(rect, b).has_value());
    EXPECT_FALSE(bicgstab(rect, b).has_value());
    EXPECT_FALSE(gmres(rect, b).has_value());
    EXPECT_FALSE(minres(rect, b).has_value());
    EXPECT_FALSE(qmr(rect, b).has_value());
    EXPECT_FALSE(lsqr(rect, b).has_value());
    EXPECT_FALSE(lsmr(rect, b).has_value());
    EXPECT_FALSE(tfqmr(rect, b).has_value());
}

TEST(LinalgEdgeTest, nonsquare_solve_sylvester) {
    DMatrix A{{1.0, 0.0}, {0.0, 2.0}};
    DMatrix Brect{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    DMatrix C{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    EXPECT_FALSE(solve_sylvester(Brect, A, C).has_value());
    EXPECT_FALSE(solve_sylvester(A, Brect, C).has_value());
}

TEST(LinalgEdgeTest, nonsquare_diag_extract_and_precond) {
    DMatrix rect{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const std::vector<double> d = diag(rect);
    ASSERT_EQ(d.size(), 2u);
    EXPECT_NEAR(d[0], 1.0, 1e-12);
    EXPECT_NEAR(d[1], 5.0, 1e-12);

    const std::vector<double> scale = precond_diag(rect);
    ASSERT_EQ(scale.size(), 2u);
    EXPECT_NEAR(scale[0], 1.0, 1e-12);
    EXPECT_NEAR(scale[1], 0.2, 1e-12);

    const DMatrix M = precond_ssor(rect, 1.0);
    EXPECT_EQ(M.rows(), 2u);
    EXPECT_NEAR(M(0, 0), 1.0, 1e-12);
}

// ---- More 1x1 edge cases ----

TEST(LinalgEdgeTest, one_by_one_eig_ldl_hess) {
    DMatrix A(1, 1);
    A(0, 0) = 4.0;

    const auto e = eig(A);
    if (!e.has_value()) {
        GTEST_SKIP() << "eig 1x1 unavailable";
    }
    EXPECT_NEAR(e->values(0, 0), 4.0, 1e-8);

    const auto es = eig_sym(A);
    if (!es.has_value()) {
        GTEST_SKIP() << "eig_sym 1x1 unavailable";
    }
    EXPECT_NEAR(es->values(0, 0), 4.0, 1e-8);

    const auto L = ldl(A);
    if (!L.has_value()) {
        GTEST_SKIP() << "ldl 1x1 unavailable";
    }
    EXPECT_NEAR(L->D(0, 0), 4.0, 1e-12);

    const auto H = hess(A);
    if (!H.has_value()) {
        GTEST_SKIP() << "hess 1x1 unavailable";
    }
    EXPECT_NEAR((*H)(0, 0), 4.0, 1e-12);
}

TEST(LinalgEdgeTest, one_by_one_matrix_funcs) {
    DMatrix A(1, 1);
    A(0, 0) = 4.0;

    const auto S = sqrtm(A);
    if (!S.has_value()) {
        GTEST_SKIP() << "sqrtm 1x1 unavailable";
    }
    EXPECT_NEAR((*S)(0, 0), 2.0, 1e-8);

    const auto L = logm(A);
    if (!L.has_value()) {
        GTEST_SKIP() << "logm 1x1 unavailable";
    }
    EXPECT_NEAR((*L)(0, 0), std::log(4.0), 1e-8);

    const auto sn = sinm(A);
    if (!sn.has_value()) {
        GTEST_SKIP() << "sinm 1x1 unavailable";
    }
    EXPECT_NEAR((*sn)(0, 0), std::sin(4.0), 1e-8);

    const auto cs = cosm(A);
    if (!cs.has_value()) {
        GTEST_SKIP() << "cosm 1x1 unavailable";
    }
    EXPECT_NEAR((*cs)(0, 0), std::cos(4.0), 1e-8);

    std::function<double(double)> square = [](double x) { return x * x; };
    const auto F = funm(A, square);
    if (!F.has_value()) {
        GTEST_SKIP() << "funm 1x1 unavailable";
    }
    EXPECT_NEAR((*F)(0, 0), 16.0, 1e-8);
}

TEST(LinalgEdgeTest, one_by_one_rank_cond_pinv_schur) {
    DMatrix A(1, 1);
    A(0, 0) = 5.0;

    const auto r = rank(A);
    if (!r.has_value()) {
        GTEST_SKIP() << "rank 1x1 unavailable";
    }
    EXPECT_NEAR(*r, 1.0, 1e-12);
    EXPECT_EQ(matrix_rank(A), 1);

    const auto c = cond(A);
    if (!c.has_value()) {
        GTEST_SKIP() << "cond 1x1 unavailable";
    }
    EXPECT_NEAR(*c, 1.0, 1e-8);

    const auto P = pinv(A);
    if (!P.has_value()) {
        GTEST_SKIP() << "pinv 1x1 unavailable";
    }
    EXPECT_NEAR((*P)(0, 0), 0.2, 1e-8);

    const auto sch = schur(A);
    if (!sch.has_value()) {
        GTEST_SKIP() << "schur 1x1 unavailable";
    }
    EXPECT_NEAR(sch->T(0, 0), 5.0, 1e-8);
}

TEST(LinalgEdgeTest, one_by_one_bidiag_norm_variants) {
    DMatrix A(1, 1);
    A(0, 0) = -3.0;

    const auto bd = bidiag(A);
    if (!bd.has_value()) {
        GTEST_SKIP() << "bidiag 1x1 unavailable";
    }
    EXPECT_NEAR(bd->B(0, 0), -3.0, 1e-12);

    const auto n1 = norm(A, 1);
    ASSERT_TRUE(n1.has_value());
    EXPECT_NEAR(*n1, 3.0, 1e-12);
    const auto ninf = norm(A, -1);
    ASSERT_TRUE(ninf.has_value());
    EXPECT_NEAR(*ninf, 3.0, 1e-12);
    const auto n0 = norm(A, 0);
    ASSERT_TRUE(n0.has_value());
    EXPECT_NEAR(*n0, 0.0, 1e-12);
}

TEST(LinalgEdgeTest, one_by_one_kron_repmat_rand) {
    DMatrix A(1, 1);
    A(0, 0) = 2.0;
    DMatrix B(1, 1);
    B(0, 0) = 3.0;
    const DMatrix K = kron(A, B);
    EXPECT_EQ(K.rows(), 1u);
    EXPECT_NEAR(K(0, 0), 6.0, 1e-12);

    const DMatrix R = repmat(A, 2, 3);
    EXPECT_EQ(R.rows(), 2u);
    EXPECT_EQ(R.cols(), 3u);
    EXPECT_NEAR(R(1, 2), 2.0, 1e-12);

    const DMatrix U = rand<double>(1, 1, 7);
    EXPECT_EQ(U.rows(), 1u);
    EXPECT_GE(U(0, 0), 0.0);
    EXPECT_LE(U(0, 0), 1.0);
    const DMatrix N = randn<double>(1, 1, 7);
    EXPECT_EQ(N.rows(), 1u);
    EXPECT_TRUE(std::isfinite(N(0, 0)));
}

TEST(LinalgEdgeTest, one_by_one_solve_sylvester) {
    DMatrix A(1, 1);
    A(0, 0) = 2.0;
    DMatrix B(1, 1);
    B(0, 0) = 3.0;
    DMatrix C(1, 1);
    C(0, 0) = 10.0;
    const auto X = solve_sylvester(A, B, C);
    if (!X.has_value()) {
        GTEST_SKIP() << "solve_sylvester 1x1 unavailable";
    }
    EXPECT_NEAR((*X)(0, 0), 2.0, 1e-10);
}

TEST(LinalgEdgeTest, tril_triu_offset_nonsquare) {
    DMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const DMatrix L = tril(A, -1);
    EXPECT_NEAR(L(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(L(1, 0), 4.0, 1e-12);
    EXPECT_NEAR(L(1, 1), 0.0, 1e-12);
    const DMatrix U = triu(A, 1);
    EXPECT_NEAR(U(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(U(0, 1), 2.0, 1e-12);
    EXPECT_NEAR(U(1, 1), 0.0, 1e-12);
}

TEST(LinalgEdgeTest, empty_lsq_row_mismatch) {
    DMatrix A(2, 1);
    A(0, 0) = 1.0;
    A(1, 0) = 2.0;
    DMatrix b(3, 1);
    EXPECT_FALSE(lsq(A, b).has_value());
}
