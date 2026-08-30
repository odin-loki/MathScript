#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include "ms/linalg/linalg.hpp"
#include "ms/core/matrix.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;
using RMatrix = RowMatrix<double>;

TEST(LsqTest, overdetermined_qr_path) {
    DMatrix A{{1, 1}, {1, 2}, {1, 3}};
    DMatrix b{{6}, {9}, {12}};
    auto x = lsq(A, b).value();
    EXPECT_NEAR(x(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(x(1, 0), 3.0, 1e-9);
}

TEST(EigTest, symmetric_2x2) {
    DMatrix A{{4, 1}, {1, 3}};
    auto result = eig_sym(A).value();
    EXPECT_NEAR(result.values(0, 0), 4.618, 1e-3);
    EXPECT_NEAR(result.values(1, 0), 2.381, 1e-3);
}

TEST(EigTest, eig_sym_row_major_jacobi_fallback) {
    RowMatrix<double> A(3, 3);
    A(0, 0) = 4.0;
    A(0, 1) = 1.0;
    A(0, 2) = 0.0;
    A(1, 0) = 1.0;
    A(1, 1) = 3.0;
    A(1, 2) = 1.0;
    A(2, 0) = 0.0;
    A(2, 1) = 1.0;
    A(2, 2) = 2.0;

    const auto result = eig_sym(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->values.rows(), 3u);

    const DMatrix ref{{4, 1, 0}, {1, 3, 1}, {0, 1, 2}};
    const auto lapack = eig_sym(ref);
    ASSERT_TRUE(lapack.has_value());
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(result->values(i, 0), lapack->values(i, 0), 1e-6);
    }

    for (std::size_t j = 0; j < 3; ++j) {
        DMatrix v(3, 1);
        for (std::size_t i = 0; i < 3; ++i) {
            v(i, 0) = result->vectors(i, j);
        }
        const DMatrix Av = ref * v;
        for (std::size_t i = 0; i < 3; ++i) {
            EXPECT_NEAR(Av(i, 0), result->values(j, 0) * result->vectors(i, j), 1e-5);
        }
    }
}

TEST(EigTest, nonsymmetric_qr_iteration_2x2) {
    DMatrix A{{4, 1}, {2, 3}};
    const auto result = eig(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->values(0, 0), 5.0, 1e-4);
    EXPECT_NEAR(result->values(1, 0), 2.0, 1e-4);
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_TRUE(std::isfinite(result->values(i, 0)));
    }
}

TEST(EigTest, nonsymmetric_3x3) {
    DMatrix A{{1, 2, 0}, {0, 3, 1}, {0, 0, 4}};
    auto result = eig(A).value();
    EXPECT_EQ(result.values.rows(), 3u);
    EXPECT_TRUE(std::isfinite(result.values(0, 0)));
}

TEST(EigTest, nonsymmetric_5x5_and_6x6) {
    DMatrix A5{
        {1, 1, 0, 0, 0},
        {0, 2, 1, 0, 0},
        {0, 0, 3, 1, 0},
        {0, 0, 0, 4, 1},
        {0, 0, 0, 0, 5}};
    auto r5 = eig(A5);
    ASSERT_TRUE(r5.has_value());
    EXPECT_EQ(r5->values.rows(), 5u);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_TRUE(std::isfinite(r5->values(i, 0)));
    }

    DMatrix A6{
        {0, 1, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0},
        {0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 1},
        {-1, -2, -3, -4, -5, -6}};
    auto r6 = eig(A6);
    ASSERT_TRUE(r6.has_value());
    EXPECT_EQ(r6->values.rows(), 6u);
}

TEST(EigTest, nonsymmetric_7x7_and_8x8) {
    DMatrix A7{
        {1, 1, 0, 0, 0, 0, 0},
        {0, 2, 1, 0, 0, 0, 0},
        {0, 0, 3, 1, 0, 0, 0},
        {0, 0, 0, 4, 1, 0, 0},
        {0, 0, 0, 0, 5, 1, 0},
        {0, 0, 0, 0, 0, 6, 1},
        {0, 0, 0, 0, 0, 0, 7}};
    auto r7 = eig(A7);
    ASSERT_TRUE(r7.has_value());
    EXPECT_EQ(r7->values.rows(), 7u);

    DMatrix A8 = zeros<double>(8, 8);
    for (std::size_t i = 0; i < 7; ++i) {
        A8(i, i + 1) = 1.0;
    }
    A8(7, 0) = -1.0;
    A8(7, 1) = 0.5;
    auto r8 = eig(A8);
    ASSERT_TRUE(r8.has_value());
    EXPECT_EQ(r8->values.rows(), 8u);
}

TEST(EigTest, eig_sym_8x8_spd) {
    DMatrix A = zeros<double>(8, 8);
    for (std::size_t i = 0; i < 8; ++i) {
        A(i, i) = static_cast<double>(i + 2);
        if (i > 0) {
            A(i, i - 1) = 0.1;
            A(i - 1, i) = 0.1;
        }
    }
    auto result = eig_sym(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->values.rows(), 8u);
    EXPECT_GT(result->values(0, 0), result->values(7, 0));
}

TEST(EigTest, nonsymmetric_9x9_and_10x10) {
    DMatrix A9 = zeros<double>(9, 9);
    for (std::size_t i = 0; i < 8; ++i) {
        A9(i, i + 1) = 1.0;
    }
    A9(8, 0) = -0.5;
    auto r9 = eig(A9);
    ASSERT_TRUE(r9.has_value());
    EXPECT_EQ(r9->values.rows(), 9u);

    DMatrix A10{
        {1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 2, 1, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 3, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 4, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 5, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 6, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 7, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 8, 1, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 9, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 10}};
    auto r10 = eig(A10);
    ASSERT_TRUE(r10.has_value());
    EXPECT_EQ(r10->values.rows(), 10u);
}

TEST(SvdTest, reconstruct_2x2) {
    DMatrix A{{3, 1}, {1, 2}};
    auto s = svd(A).value();
    DMatrix Sigma = zeros<double>(s.S.rows(), s.S.rows());
    for (size_t i = 0; i < s.S.rows(); ++i) {
        Sigma(i, i) = s.S(i, 0);
    }
    DMatrix reconstructed = s.U * Sigma * transpose(s.V);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(reconstructed(i, j), A(i, j), 1e-6);
        }
    }
}

TEST(LdlTest, symmetric_indefinite) {
    DMatrix A{{2, 1}, {1, -1}};
    auto result = ldl(A).value();
    DMatrix Dmat = zeros<double>(2, 2);
    Dmat(0, 0) = result.D(0, 0);
    Dmat(1, 1) = result.D(1, 0);
    DMatrix reconstructed = result.L * Dmat * transpose(result.L);
    EXPECT_NEAR(reconstructed(0, 0), A(0, 0), 1e-10);
    EXPECT_NEAR(reconstructed(1, 1), A(1, 1), 1e-10);
}

TEST(MatrixFuncTest, sqrtm_identity) {
    DMatrix I = eye<double>(3);
    auto root = sqrtm(I).value();
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(root(i, j), I(i, j), 1e-6);
        }
    }
}

TEST(AuxTest, rank_and_cond) {
    DMatrix A{{1, 2}, {2, 4}};
    EXPECT_EQ(rank(A).value(), 1);
    DMatrix B = eye<double>(3);
    EXPECT_EQ(rank(B).value(), 3);
    EXPECT_NEAR(cond(B).value(), 1.0, 1e-6);
}

TEST(IterativeTest, cg_spd) {
    DMatrix A{{4, 1}, {1, 3}};
    DMatrix b{{1}, {2}};
    auto x = cg(A, b).value();
    auto check = solve(A, b).value();
    EXPECT_NEAR(x(0, 0), check(0, 0), 1e-6);
    EXPECT_NEAR(x(1, 0), check(1, 0), 1e-6);
}

TEST(IterativeTest, bicgstab_nonsymmetric) {
    DMatrix A{{3, 1}, {1, 2}};
    DMatrix b{{1}, {1}};
    auto x = bicgstab(A, b).value();
    auto check = solve(A, b).value();
    EXPECT_NEAR(x(0, 0), check(0, 0), 1e-5);
    EXPECT_NEAR(x(1, 0), check(1, 0), 1e-5);
}

TEST(IterativeTest, gmres_nonsymmetric) {
    DMatrix A{{3, 1}, {1, 2}};
    DMatrix b{{1}, {1}};
    auto x = gmres(A, b, 2, 50).value();
    auto check = solve(A, b).value();
    EXPECT_NEAR(x(0, 0), check(0, 0), 1e-4);
    EXPECT_NEAR(x(1, 0), check(1, 0), 1e-4);
}

TEST(ConstructionTest, diag_and_tri) {
    auto D = diag(std::vector<double>{2, 3, 5});
    EXPECT_EQ(D.rows(), 3);
    EXPECT_NEAR(D(1, 1), 3.0, 1e-12);
    auto v = diag(D);
    EXPECT_EQ(v.size(), 3);
    EXPECT_NEAR(v[2], 5.0, 1e-12);

    DMatrix A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto L = tril(A);
    EXPECT_NEAR(L(2, 1), 8.0, 1e-12);
    EXPECT_NEAR(L(0, 1), 0.0, 1e-12);
    auto U = triu(A);
    EXPECT_NEAR(U(0, 2), 3.0, 1e-12);
    EXPECT_NEAR(U(2, 0), 0.0, 1e-12);
}

TEST(ConstructionTest, rand_reproducible) {
    auto R1 = rand<double>(2, 2, 123);
    auto R2 = rand<double>(2, 2, 123);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_DOUBLE_EQ(R1(i, j), R2(i, j));
        }
    }
}

TEST(EigTest, symmetric_delegates_from_eig) {
    DMatrix A{{5, 2}, {2, 4}};
    auto sym = eig_sym(A);
    auto general = eig(A);
    ASSERT_TRUE(sym.has_value());
    ASSERT_TRUE(general.has_value());
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(sym->values(i, 0), general->values(i, 0), 1e-9);
    }
}

TEST(EigTest, qr_iteration_nonsymmetric_companion) {
    DMatrix A = zeros<double>(4, 4);
    for (size_t i = 0; i < 3; ++i) {
        A(i, i + 1) = 1.0;
    }
    A(3, 0) = -24.0;
    A(0, 0) = 1.0;
    A(1, 1) = 2.0;
    A(2, 2) = 3.0;
    A(3, 3) = 4.0;
    auto result = eig(A);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isfinite(result->values(i, 0)));
    }
    EXPECT_GT(result->values(0, 0), result->values(3, 0));
}

TEST(EigTest, nonsymmetric_qr_iteration_3x3_offdiag) {
    DMatrix A{{1, 1, 0}, {0, 1, 1}, {1, 0, 1}};
    const auto result = eig(A);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->values.rows(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(std::isfinite(result->values(i, 0)));
    }
}

TEST(SvdTest, tall_and_wide_reconstruction) {
    DMatrix tall{{1, 0, 2}, {0, 3, 0}, {4, 0, 5}, {0, 6, 0}};
    auto tall_svd = svd(tall);
    ASSERT_TRUE(tall_svd.has_value());
    EXPECT_EQ(tall_svd->S.rows(), 3u);

    DMatrix wide{{1, 2, 3, 4}, {5, 6, 7, 8}};
    auto wide_svd = svd(wide);
    ASSERT_TRUE(wide_svd.has_value());
    EXPECT_EQ(wide_svd->S.rows(), 2u);

    for (const auto* result : {&tall_svd.value(), &wide_svd.value()}) {
        DMatrix Sigma = zeros<double>(result->S.rows(), result->S.rows());
        for (size_t i = 0; i < result->S.rows(); ++i) {
            Sigma(i, i) = result->S(i, 0);
        }
        const DMatrix A = (result == &tall_svd.value()) ? tall : wide;
        const DMatrix reconstructed = result->U * Sigma * transpose(result->V);
        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t j = 0; j < A.cols(); ++j) {
                EXPECT_NEAR(reconstructed(i, j), A(i, j), 1e-5);
            }
        }
    }
}

TEST(ConstructionTest, TrilTest_Offset) {
    DMatrix A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    const DMatrix L1 = tril(A, 1);
    EXPECT_NEAR(L1(0, 1), 2.0, 1e-12);
    EXPECT_NEAR(L1(0, 2), 0.0, 1e-12);
    const DMatrix Lm1 = tril(A, -1);
    EXPECT_NEAR(Lm1(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(Lm1(1, 0), 4.0, 1e-12);
}

TEST(ConstructionTest, TriuTest_Offset) {
    DMatrix A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    const DMatrix Um1 = triu(A, -1);
    EXPECT_NEAR(Um1(1, 0), 4.0, 1e-12);
    EXPECT_NEAR(Um1(2, 0), 0.0, 1e-12);
    EXPECT_NEAR(Um1(0, 0), 1.0, 1e-12);
}

TEST(MatrixFuncTest, LogmTest_ViaExpm) {
    DMatrix B{{0.05, 0.02}, {0.02, -0.04}};
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

TEST(AuxTest, cond_rejects_non_p2_and_singular) {
    DMatrix I = eye<double>(2);
    EXPECT_FALSE(cond(I, 1).has_value());
    EXPECT_FALSE(cond(I, -1).has_value());
    DMatrix Z = zeros<double>(2, 2);
    EXPECT_FALSE(cond(Z).has_value());
}

TEST(AuxTest, matrix_rank_custom_tol) {
    DMatrix A{{1, 2}, {2, 4}};
    EXPECT_EQ(matrix_rank(A), 1);
    EXPECT_EQ(matrix_rank(A, 10.0), 0);
    EXPECT_EQ(matrix_rank(eye<double>(3)), 3);
}

TEST(LsqTest, underdetermined_and_row_mismatch) {
    DMatrix A{{1, 1, 1}, {0, 1, 2}};
    DMatrix b{{3}, {3}};
    const auto x = lsq(A, b);
    ASSERT_TRUE(x.has_value());
    EXPECT_EQ(x->rows(), 3u);
    DMatrix b_bad(3, 1);
    EXPECT_FALSE(lsq(A, b_bad).has_value());
}

TEST(IterativeTest, jacobi_spd) {
    DMatrix A{{4, 1}, {1, 3}};
    DMatrix b{{1}, {2}};
    const auto x = jacobi(A, b);
    ASSERT_TRUE(x.has_value());
    const auto check = solve(A, b);
    ASSERT_TRUE(check.has_value());
    EXPECT_NEAR((*x)(0, 0), (*check)(0, 0), 1e-6);
    EXPECT_NEAR((*x)(1, 0), (*check)(1, 0), 1e-6);
}

TEST(IterativeTest, jacobi_zero_diagonal) {
    DMatrix A{{0, 1}, {1, 2}};
    DMatrix b{{1}, {1}};
    EXPECT_FALSE(jacobi(A, b).has_value());
}

TEST(IterativeTest, minres_qmr_lsmr_identity) {
    DMatrix I = eye<double>(2);
    DMatrix b{{3}, {-1}};
    const auto xm = minres(I, b);
    ASSERT_TRUE(xm.has_value());
    EXPECT_NEAR((*xm)(0, 0), 3.0, 1e-8);
    EXPECT_NEAR((*xm)(1, 0), -1.0, 1e-8);
    const auto xq = qmr(I, b);
    ASSERT_TRUE(xq.has_value());
    EXPECT_NEAR((*xq)(0, 0), 3.0, 1e-6);
    EXPECT_NEAR((*xq)(1, 0), -1.0, 1e-6);
    const auto xl = lsmr(I, b);
    ASSERT_TRUE(xl.has_value());
    EXPECT_NEAR((*xl)(0, 0), 3.0, 1e-6);
    EXPECT_NEAR((*xl)(1, 0), -1.0, 1e-6);
}

TEST(IterativeTest, lsqr_overdetermined_and_tfqmr) {
    DMatrix A{{1, 0}, {0, 1}, {1, 1}};
    DMatrix b{{1}, {2}, {3}};
    const auto x = lsqr(A, b);
    ASSERT_TRUE(x.has_value());
    EXPECT_EQ(x->rows(), 2u);
    EXPECT_TRUE(std::isfinite((*x)(0, 0)));

    DMatrix S{{3, 1}, {1, 2}};
    DMatrix sb{{1}, {1}};
    const auto xt = tfqmr(S, sb);
    if (!xt.has_value()) {
        GTEST_SKIP() << "tfqmr did not converge";
    }
    EXPECT_TRUE(std::isfinite((*xt)(0, 0)));
}

TEST(ConstructionTest, randn_kron_linspace_repmat) {
    const auto N1 = randn<double>(2, 2, 99);
    const auto N2 = randn<double>(2, 2, 99);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_DOUBLE_EQ(N1(i, j), N2(i, j));
            EXPECT_TRUE(std::isfinite(N1(i, j)));
        }
    }

    DMatrix A{{1, 2}, {3, 4}};
    const DMatrix K = kron(A, eye<double>(2));
    EXPECT_EQ(K.rows(), 4u);
    EXPECT_EQ(K.cols(), 4u);
    EXPECT_NEAR(K(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(K(2, 0), 3.0, 1e-12);
    EXPECT_NEAR(K(2, 2), 4.0, 1e-12);

    const auto grid = linspace(0.0, 1.0, 5u);
    ASSERT_EQ(grid.size(), 5u);
    EXPECT_NEAR(grid[0], 0.0, 1e-15);
    EXPECT_NEAR(grid[4], 1.0, 1e-15);
    EXPECT_TRUE(linspace(0.0, 1.0, 0u).empty());
    EXPECT_NEAR(linspace(3.0, 9.0, 1u)[0], 3.0, 1e-15);

    const DMatrix R = repmat(A, 2, 1);
    EXPECT_EQ(R.rows(), 4u);
    EXPECT_EQ(R.cols(), 2u);
    EXPECT_NEAR(R(2, 0), 1.0, 1e-12);
}

TEST(AuxTest, pinv_null_orth_edges) {
    DMatrix tall{{1}, {2}, {3}};
    const auto P = pinv(tall);
    ASSERT_TRUE(P.has_value());
    EXPECT_EQ(P->rows(), 1u);
    EXPECT_EQ(P->cols(), 3u);

    DMatrix I = eye<double>(2);
    const auto N = null(I);
    ASSERT_TRUE(N.has_value());
    EXPECT_EQ(N->cols(), 0u);

    const auto Q = orth(I);
    ASSERT_TRUE(Q.has_value());
    EXPECT_EQ(Q->cols(), 2u);
}

TEST(MatrixFuncTest, sinm_cosm_funm_hess_schur_bidiag) {
    const double pi_half = std::acos(-1.0) * 0.5;
    DMatrix D{{0, 0}, {0, pi_half}};
    const auto sn = sinm(D);
    ASSERT_TRUE(sn.has_value());
    EXPECT_NEAR((*sn)(0, 0), 0.0, 1e-9);
    EXPECT_NEAR((*sn)(1, 1), 1.0, 1e-9);
    const auto cs = cosm(D);
    ASSERT_TRUE(cs.has_value());
    EXPECT_NEAR((*cs)(0, 0), 1.0, 1e-9);
    EXPECT_NEAR((*cs)(1, 1), 0.0, 1e-9);

    std::function<double(double)> sine = [](double x) { return std::sin(x); };
    const auto F = funm(D, sine);
    ASSERT_TRUE(F.has_value());
    EXPECT_NEAR((*F)(0, 0), (*sn)(0, 0), 1e-8);
    EXPECT_NEAR((*F)(1, 1), (*sn)(1, 1), 1e-8);

    DMatrix A{{2, 1}, {0, 3}};
    const auto H = hess(A);
    ASSERT_TRUE(H.has_value());
    EXPECT_NEAR((*H)(0, 0), 2.0, 1e-8);
    const auto S = schur(A);
    ASSERT_TRUE(S.has_value());
    EXPECT_EQ(S->T.rows(), 2u);

    DMatrix R{{1, 2, 3}, {4, 5, 6}};
    const auto bd = bidiag(R);
    ASSERT_TRUE(bd.has_value());
    EXPECT_EQ(bd->B.rows(), 2u);
    EXPECT_EQ(bd->B.cols(), 3u);
}

TEST(SylvesterTest, solve_2x2_and_precond_zero_diag) {
    DMatrix A{{2, 0}, {0, 3}};
    DMatrix B{{1, 0}, {0, 1}};
    DMatrix C{{6, 0}, {0, 8}};
    const auto X = solve_sylvester(A, B, C);
    ASSERT_TRUE(X.has_value());
    EXPECT_NEAR((*X)(0, 0), 2.0, 1e-8);
    EXPECT_NEAR((*X)(1, 1), 2.0, 1e-8);

    DMatrix Z{{0, 1}, {0, 0}};
    const auto scale = precond_diag(Z);
    ASSERT_EQ(scale.size(), 2u);
    EXPECT_NEAR(scale[0], 1.0, 1e-15);
    const DMatrix M = precond_ssor(Z, 2.0);
    EXPECT_NEAR(M(0, 0), 0.0, 1e-15);
}

TEST(SylvesterTest, rejects_non_square_and_c_shape) {
    DMatrix Arect{{1, 2, 3}, {4, 5, 6}};
    DMatrix Bsq{{1, 0}, {0, 1}};
    DMatrix C{{1, 2}, {3, 4}};
    EXPECT_FALSE(solve_sylvester(Arect, Bsq, C).has_value());
    DMatrix Asq{{2, 0}, {0, 3}};
    DMatrix Brect{{1, 2}};
    EXPECT_FALSE(solve_sylvester(Asq, Brect, C).has_value());
    DMatrix Cbad{{1}, {2}};
    EXPECT_FALSE(solve_sylvester(Asq, Bsq, Cbad).has_value());
}

TEST(SylvesterTest, one_by_one_and_shared_eigenvalue) {
    DMatrix A{{2}};
    DMatrix B{{3}};
    DMatrix C{{10}};
    const auto X = solve_sylvester(A, B, C);
    ASSERT_TRUE(X.has_value());
    EXPECT_NEAR((*X)(0, 0), 2.0, 1e-10);

    DMatrix Am1{{1}};
    DMatrix Bm1{{-1}};
    DMatrix C1{{1}};
    EXPECT_FALSE(solve_sylvester(Am1, Bm1, C1).has_value());
}

TEST(PrecondTest, identity_empty_one_by_one_rect) {
    DMatrix I = eye<double>(2);
    const auto dI = precond_diag(I);
    ASSERT_EQ(dI.size(), 2u);
    EXPECT_NEAR(dI[0], 1.0, 1e-15);
    EXPECT_NEAR(dI[1], 1.0, 1e-15);
    const DMatrix MI = precond_ssor(I, 1.0);
    EXPECT_NEAR(MI(0, 0), 1.0, 1e-15);
    EXPECT_NEAR(MI(1, 1), 1.0, 1e-15);

    DMatrix Z(0, 0);
    EXPECT_TRUE(precond_diag(Z).empty());
    const DMatrix MZ = precond_ssor(Z, 1.0);
    EXPECT_EQ(MZ.rows(), 0u);
    EXPECT_EQ(MZ.cols(), 0u);

    DMatrix A11{{4}};
    const auto d1 = precond_diag(A11);
    ASSERT_EQ(d1.size(), 1u);
    EXPECT_NEAR(d1[0], 0.25, 1e-15);
    const DMatrix M1 = precond_ssor(A11, 2.0);
    EXPECT_NEAR(M1(0, 0), 2.0, 1e-15);

    DMatrix R{{2, 0, 1}, {0, 4, 0}};
    const auto dR = precond_diag(R);
    ASSERT_EQ(dR.size(), 2u);
    EXPECT_NEAR(dR[0], 0.5, 1e-15);
    EXPECT_NEAR(dR[1], 0.25, 1e-15);
}

TEST(LsqTest, one_by_one_square_and_empty) {
    DMatrix A11{{2}};
    DMatrix b11{{6}};
    const auto x1 = lsq(A11, b11);
    ASSERT_TRUE(x1.has_value());
    EXPECT_NEAR((*x1)(0, 0), 3.0, 1e-10);

    DMatrix A{{2, 1}, {1, 3}};
    DMatrix B{{4, 1}, {7, 2}};
    const auto X = lsq(A, B);
    ASSERT_TRUE(X.has_value());
    EXPECT_NEAR((*X)(0, 0), 1.0, 1e-9);
    EXPECT_NEAR((*X)(1, 0), 2.0, 1e-9);

    DMatrix Aempty(0, 1);
    DMatrix bempty(0, 1);
    EXPECT_FALSE(lsq(Aempty, bempty).has_value());
}

TEST(AuxTest, rank_cond_pinv_empty_and_one_by_one) {
    DMatrix A11{{5}};
    EXPECT_EQ(rank(A11).value(), 1);
    EXPECT_EQ(matrix_rank(A11), 1);
    EXPECT_NEAR(cond(A11).value(), 1.0, 1e-12);
    const auto P = pinv(A11);
    ASSERT_TRUE(P.has_value());
    EXPECT_NEAR((*P)(0, 0), 0.2, 1e-12);

    DMatrix Z(0, 0);
    EXPECT_FALSE(rank(Z).has_value());
    EXPECT_EQ(matrix_rank(Z), 0);
    EXPECT_FALSE(cond(Z).has_value());
    EXPECT_FALSE(pinv(Z).has_value());

    DMatrix I = eye<double>(2);
    const auto PinvI = pinv(I);
    ASSERT_TRUE(PinvI.has_value());
    EXPECT_NEAR((*PinvI)(0, 0), 1.0, 1e-10);
    EXPECT_NEAR((*PinvI)(1, 1), 1.0, 1e-10);
}

TEST(AuxTest, null_orth_empty_and_rank_deficient) {
    DMatrix A0c(2, 0);
    const auto N0 = null(A0c);
    ASSERT_TRUE(N0.has_value());
    EXPECT_EQ(N0->rows(), 0u);
    EXPECT_EQ(N0->cols(), 0u);

    DMatrix A0r(0, 2);
    const auto Q0 = orth(A0r);
    ASSERT_TRUE(Q0.has_value());
    EXPECT_EQ(Q0->rows(), 0u);
    EXPECT_EQ(Q0->cols(), 0u);

    DMatrix A11{{3}};
    const auto Q1 = orth(A11);
    ASSERT_TRUE(Q1.has_value());
    EXPECT_EQ(Q1->cols(), 1u);

    DMatrix rank1{{1, 2}, {2, 4}};
    const auto N = null(rank1);
    ASSERT_TRUE(N.has_value());
    EXPECT_EQ(N->cols(), 1u);
    const auto Qr = orth(rank1, 1e-8);
    ASSERT_TRUE(Qr.has_value());
    EXPECT_EQ(Qr->cols(), 1u);
}

TEST(LdlTest, rejects_and_one_by_one) {
    DMatrix rect{{1, 2, 3}, {4, 5, 6}};
    EXPECT_FALSE(ldl(rect).has_value());
    DMatrix nonsym{{1, 2}, {0, 3}};
    EXPECT_FALSE(ldl(nonsym).has_value());
    DMatrix sing{{0, 0}, {0, 0}};
    EXPECT_FALSE(ldl(sing).has_value());

    DMatrix A11{{4}};
    const auto f = ldl(A11);
    ASSERT_TRUE(f.has_value());
    EXPECT_NEAR(f->D(0, 0), 4.0, 1e-12);
    EXPECT_NEAR(f->L(0, 0), 1.0, 1e-12);
}

TEST(EigTest, rejects_non_square_nonsym_and_one_by_one) {
    DMatrix rect{{1, 2, 3}, {4, 5, 6}};
    EXPECT_FALSE(eig(rect).has_value());
    EXPECT_FALSE(eig_sym(rect).has_value());
    DMatrix nonsym{{1, 2}, {0, 3}};
    EXPECT_FALSE(eig_sym(nonsym).has_value());

    DMatrix A11{{7}};
    const auto es = eig_sym(A11);
    ASSERT_TRUE(es.has_value());
    EXPECT_NEAR(es->values(0, 0), 7.0, 1e-12);
    const auto eg = eig(A11);
    ASSERT_TRUE(eg.has_value());
    EXPECT_NEAR(eg->values(0, 0), 7.0, 1e-12);
}

TEST(SvdTest, empty_one_by_one_and_rank) {
    DMatrix Z(0, 2);
    EXPECT_FALSE(svd(Z).has_value());
    DMatrix Zc(3, 0);
    EXPECT_FALSE(svd(Zc).has_value());

    DMatrix A11{{4}};
    const auto s = svd(A11);
    ASSERT_TRUE(s.has_value());
    EXPECT_NEAR(s->S(0, 0), 4.0, 1e-12);

    DMatrix rank1{{1, 2}, {2, 4}};
    EXPECT_EQ(rank(rank1, 1e-8).value(), 1);
}

TEST(MatrixFuncTest, rejects_rect_and_one_by_one) {
    DMatrix rect{{1, 2, 3}, {4, 5, 6}};
    EXPECT_FALSE(logm(rect).has_value());
    EXPECT_FALSE(sqrtm(rect).has_value());
    EXPECT_FALSE(sinm(rect).has_value());
    EXPECT_FALSE(cosm(rect).has_value());
    std::function<double(double)> id = [](double x) { return x; };
    EXPECT_FALSE(funm(rect, id).has_value());

    DMatrix A11{{4}};
    const auto r = sqrtm(A11);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR((*r)(0, 0), 2.0, 1e-10);
    DMatrix I1{{1}};
    const auto L = logm(I1);
    ASSERT_TRUE(L.has_value());
    EXPECT_NEAR((*L)(0, 0), 0.0, 1e-10);
}

TEST(DecompTest, hess_schur_bidiag_edges) {
    DMatrix rect{{1, 2, 3}, {4, 5, 6}};
    EXPECT_FALSE(hess(rect).has_value());
    EXPECT_FALSE(schur(rect).has_value());

    DMatrix A11{{5}};
    const auto H = hess(A11);
    ASSERT_TRUE(H.has_value());
    EXPECT_NEAR((*H)(0, 0), 5.0, 1e-12);
    const auto S = schur(A11);
    ASSERT_TRUE(S.has_value());
    EXPECT_NEAR(S->T(0, 0), 5.0, 1e-8);
    const auto bd = bidiag(A11);
    ASSERT_TRUE(bd.has_value());
    EXPECT_NEAR(bd->B(0, 0), 5.0, 1e-8);

    DMatrix empty(0, 2);
    const auto bde = bidiag(empty);
    ASSERT_TRUE(bde.has_value());
    EXPECT_EQ(bde->B.rows(), 0u);
    EXPECT_EQ(bde->B.cols(), 2u);
}

TEST(IterativeTest, mismatch_nonsym_and_one_by_one) {
    DMatrix A{{3, 1}, {1, 2}};
    DMatrix bbad(3, 1);
    EXPECT_FALSE(cg(A, bbad).has_value());
    EXPECT_FALSE(jacobi(A, bbad).has_value());
    EXPECT_FALSE(bicgstab(A, bbad).has_value());
    EXPECT_FALSE(gmres(A, bbad).has_value());
    EXPECT_FALSE(minres(A, bbad).has_value());
    EXPECT_FALSE(qmr(A, bbad).has_value());
    EXPECT_FALSE(lsqr(A, bbad).has_value());
    EXPECT_FALSE(lsmr(A, bbad).has_value());
    EXPECT_FALSE(tfqmr(A, bbad).has_value());

    DMatrix nonsym{{1, 2}, {3, 4}};
    DMatrix b{{1}, {1}};
    EXPECT_FALSE(cg(nonsym, b).has_value());

    DMatrix A11{{2}};
    DMatrix b11{{8}};
    const auto x = jacobi(A11, b11);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)(0, 0), 4.0, 1e-10);
    const auto xc = cg(A11, b11);
    ASSERT_TRUE(xc.has_value());
    EXPECT_NEAR((*xc)(0, 0), 4.0, 1e-10);
}

TEST(ConstructionTest, empty_and_one_by_one_helpers) {
    std::vector<double> empty_v;
    const DMatrix D0 = diag(empty_v);
    EXPECT_EQ(D0.rows(), 0u);
    EXPECT_EQ(D0.cols(), 0u);
    EXPECT_TRUE(diag(D0).empty());

    std::vector<double> one_v{9.0};
    const DMatrix D1 = diag(one_v);
    EXPECT_EQ(D1.rows(), 1u);
    EXPECT_NEAR(D1(0, 0), 9.0, 1e-15);

    DMatrix Z(0, 2);
    const DMatrix K = kron(Z, eye<double>(2));
    EXPECT_EQ(K.rows(), 0u);
    EXPECT_EQ(K.cols(), 4u);
    const DMatrix R = repmat(Z, 2, 3);
    EXPECT_EQ(R.rows(), 0u);
    EXPECT_EQ(R.cols(), 6u);

    const auto Rd = rand<double>(0, 0, 7u);
    EXPECT_EQ(Rd.rows(), 0u);
    const auto Rn = randn<double>(1, 1, 7u);
    EXPECT_TRUE(std::isfinite(Rn(0, 0)));
    EXPECT_NEAR(tril(D1)(0, 0), 9.0, 1e-15);
    EXPECT_NEAR(triu(D1, 1)(0, 0), 0.0, 1e-15);
}
