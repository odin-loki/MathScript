#include <gtest/gtest.h>
#include "ms/linalg/linalg.hpp"

using namespace ms;

TEST(ConstructionTest, rand_shape) {
    auto M = rand<double>(3, 4, 42);
    EXPECT_EQ(M.rows(), 3u);
    EXPECT_EQ(M.cols(), 4u);
}

TEST(ConstructionTest, rand_range) {
    auto M = rand<double>(10, 10, 7);
    for (size_t i = 0; i < M.rows(); ++i) {
        for (size_t j = 0; j < M.cols(); ++j) {
            EXPECT_GE(M(i, j), 0.0);
            EXPECT_LT(M(i, j), 1.0);
        }
    }
}

TEST(ConstructionTest, randn_shape) {
    auto M = randn<double>(2, 5, 0);
    EXPECT_EQ(M.rows(), 2u);
    EXPECT_EQ(M.cols(), 5u);
}

TEST(ConstructionTest, rand_different_seeds) {
    auto A = rand<double>(3, 3, 1);
    auto B = rand<double>(3, 3, 2);
    bool any_different = false;
    for (size_t i = 0; i < A.rows() && !any_different; ++i) {
        for (size_t j = 0; j < A.cols() && !any_different; ++j) {
            if (A(i, j) != B(i, j)) {
                any_different = true;
            }
        }
    }
    EXPECT_TRUE(any_different);
}

TEST(ConstructionTest, rand_same_seed) {
    auto A = rand<double>(3, 3, 42);
    auto B = rand<double>(3, 3, 42);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_DOUBLE_EQ(A(i, j), B(i, j));
        }
    }
}
