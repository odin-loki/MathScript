#include <gtest/gtest.h>
#include <cmath>
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
