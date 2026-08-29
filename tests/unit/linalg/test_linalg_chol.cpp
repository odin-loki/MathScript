#include <gtest/gtest.h>

#include "ms/linalg/linalg.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

TEST(CholExtTest, spd_factorization) {
    DMatrix S{{4, 1}, {1, 3}};
    const auto L = chol(S).value();
    const DMatrix recon = L * transpose(L);
    EXPECT_NEAR(recon(0, 0), S(0, 0), 1e-9);
    EXPECT_NEAR(recon(1, 1), S(1, 1), 1e-9);
}

TEST(CholExtTest, rejects_non_spd) {
    DMatrix A{{1, 2}, {3, 4}};
    EXPECT_FALSE(chol(A).has_value());
}

TEST(CholExtTest, float_generic_path) {
    ColMatrix<float> S{{4.f, 1.f}, {1.f, 3.f}};
    const auto L = chol(S);
    ASSERT_TRUE(L.has_value());
    EXPECT_GT((*L)(0, 0), 0.f);
    EXPECT_GT((*L)(1, 1), 0.f);
}

TEST(ConstructionExtTest, randn_and_diag) {
    const auto R1 = randn<double>(2, 2, 77);
    const auto R2 = randn<double>(2, 2, 77);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_DOUBLE_EQ(R1(i, j), R2(i, j));
        }
    }
    const auto D = diag(std::vector<double>{1, 4, 9});
    EXPECT_NEAR(D(2, 2), 9.0, 1e-12);
}
