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
