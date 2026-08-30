#include <gtest/gtest.h>

#include "ms/linalg/linalg.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

TEST(IterativeExtTest, cg_rejects_nonsymmetric) {
    DMatrix A{{1, 2}, {3, 4}};
    DMatrix b{{1}, {1}};
    EXPECT_FALSE(cg(A, b).has_value());
}

TEST(IterativeExtTest, dimension_mismatch) {
    DMatrix A{{4, 1}, {1, 3}};
    DMatrix b{{1}, {2}, {3}};
    EXPECT_FALSE(cg(A, b).has_value());
}

TEST(IterativeExtTest, bicgstab_converges_with_defaults) {
    DMatrix A{{3, 1}, {1, 2}};
    DMatrix b{{1}, {1}};
    const auto x = bicgstab(A, b).value();
    const auto ref = solve(A, b).value();
    EXPECT_NEAR(x(0, 0), ref(0, 0), 1e-5);
    EXPECT_NEAR(x(1, 0), ref(1, 0), 1e-5);
}

TEST(QrWideTest, underdetermined_modified_gram_schmidt) {
    DMatrix A{{1, 2, 3}, {4, 5, 6}};
    const auto [Q, R] = qr(A).value();
    EXPECT_EQ(Q.rows(), 2u);
    EXPECT_EQ(R.cols(), 3u);
    const DMatrix recon = Q * R;
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(recon(i, j), A(i, j), 1e-6);
        }
    }
}

TEST(IterativeExtTest, GmresTest_NonConvergent) {
    // 3x3 tridiagonal: 1 GMRES step cannot reduce residual to 1e-15
    DMatrix A{{10, 1, 0}, {1, 10, 1}, {0, 1, 10}};
    DMatrix b{{1}, {2}, {3}};
    const auto x = gmres(A, b, 1, 1, 1e-15);
    EXPECT_FALSE(x.has_value());
}

TEST(IterativeExtTest, BicgstabTest_DimMismatch) {
    DMatrix A{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    DMatrix b{{1}, {2}};
    EXPECT_FALSE(bicgstab(A, b).has_value());
}

TEST(IterativeExtTest, CgTest_MaxIterExceeded) {
    DMatrix A{{4, 1, 0}, {1, 4, 1}, {0, 1, 4}};
    DMatrix b{{1}, {2}, {3}};
    const auto x = cg(A, b, 1, 1e-15);
    EXPECT_FALSE(x.has_value());
}

TEST(IterativeExtTest, RankTest_CustomTolerance) {
    DMatrix A{{1, 0}, {0, 1e-5}};
    const auto r_default = rank(A, 0.0);
    const auto r_tol = rank(A, 0.1);
    ASSERT_TRUE(r_default.has_value());
    ASSERT_TRUE(r_tol.has_value());
    EXPECT_EQ(static_cast<size_t>(*r_default), 2u);
    EXPECT_EQ(static_cast<size_t>(*r_tol), 1u);
}

TEST(IterativeExtTest, JacobiTest_EmptyAndMismatch) {
    DMatrix empty_A(0, 0);
    DMatrix rhs_one{{1.0}};
    EXPECT_FALSE(jacobi(empty_A, rhs_one).has_value());

    DMatrix rect{{1.0, 2.0}};
    DMatrix b_rect{{1.0}};
    EXPECT_FALSE(jacobi(rect, b_rect).has_value());
}

TEST(IterativeExtTest, MinresTest_EmptyAndMismatch) {
    DMatrix empty_A(0, 0);
    DMatrix rhs_one{{1.0}};
    EXPECT_FALSE(minres(empty_A, rhs_one).has_value());

    DMatrix rect{{1.0, 2.0, 3.0}};
    DMatrix b_rect{{1.0}};
    EXPECT_FALSE(minres(rect, b_rect).has_value());
}

TEST(IterativeExtTest, QmrTest_EmptyAndMismatch) {
    DMatrix empty_A(0, 0);
    DMatrix rhs_one{{1.0}};
    EXPECT_FALSE(qmr(empty_A, rhs_one).has_value());

    DMatrix tall_A{{1.0}, {2.0}};
    DMatrix short_b{{1.0}};
    EXPECT_FALSE(qmr(tall_A, short_b).has_value());
}

TEST(IterativeExtTest, LsqrTest_EmptyAndMismatch) {
    DMatrix empty_A(0, 0);
    DMatrix rhs_one{{1.0}};
    EXPECT_FALSE(lsqr(empty_A, rhs_one).has_value());

    DMatrix A{{1.0, 0.0}, {0.0, 1.0}};
    DMatrix tall_b{{1.0}, {2.0}, {3.0}};
    EXPECT_FALSE(lsqr(A, tall_b).has_value());
}

TEST(IterativeExtTest, GmresTest_EmptyAndMismatch) {
    DMatrix empty_A(0, 0);
    DMatrix rhs_one{{1.0}};
    EXPECT_FALSE(gmres(empty_A, rhs_one).has_value());

    DMatrix ident{{1.0, 0.0}, {0.0, 1.0}};
    DMatrix tall_b{{1.0}, {2.0}, {3.0}};
    EXPECT_FALSE(gmres(ident, tall_b).has_value());
}

TEST(IterativeExtTest, CgTest_EmptyMatrix) {
    DMatrix empty_A(0, 0);
    DMatrix rhs_one{{1.0}};
    EXPECT_FALSE(cg(empty_A, rhs_one).has_value());
}

TEST(IterativeExtTest, BicgstabTest_EmptyMatrix) {
    DMatrix empty_A(0, 0);
    DMatrix rhs_one{{1.0}};
    EXPECT_FALSE(bicgstab(empty_A, rhs_one).has_value());
}

TEST(IterativeExtTest, TfqmrTest_DimMismatch) {
    DMatrix A{{1.0, 0.0}, {0.0, 1.0}};
    DMatrix tall_b{{1.0}, {2.0}, {3.0}};
    EXPECT_FALSE(tfqmr(A, tall_b).has_value());
}

TEST(IterativeExtTest, LsmrTest_DimMismatch) {
    DMatrix A{{1.0, 0.0}, {0.0, 1.0}};
    DMatrix tall_b{{1.0}, {2.0}, {3.0}};
    EXPECT_FALSE(lsmr(A, tall_b).has_value());
}

TEST(IterativeExtTest, JacobiTest_ZeroDiagonal) {
    DMatrix A{{0.0, 1.0}, {1.0, 0.0}};
    DMatrix b{{1.0}, {1.0}};
    EXPECT_FALSE(jacobi(A, b).has_value());
}

TEST(IterativeExtTest, JacobiTest_MaxIterExceeded) {
    DMatrix A{{4.0, 1.0, 0.0}, {1.0, 4.0, 1.0}, {0.0, 1.0, 4.0}};
    DMatrix b{{1.0}, {2.0}, {3.0}};
    EXPECT_FALSE(jacobi(A, b, 1, 1e-15).has_value());
}
