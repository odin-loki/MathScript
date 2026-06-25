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
