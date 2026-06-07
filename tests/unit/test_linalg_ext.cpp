#include <gtest/gtest.h>
#include "ms/linalg/linalg.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

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
