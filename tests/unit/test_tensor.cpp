#include <gtest/gtest.h>
#include "ms/core/tensor.hpp"

using namespace ms;

TEST(TensorTest, 2d_construction) {
    Tensor<double, 2> t(3, 4);
    EXPECT_EQ(t.dims(), 2u);
    EXPECT_EQ(t.size(0), 3u);
    EXPECT_EQ(t.size(1), 4u);
}

TEST(TensorTest, 3d_construction) {
    // Use 3-arg constructor on Tensor<double,2> which supports dim3 > 0
    Tensor<double, 2> t(2, 3, 4);
    EXPECT_EQ(t.dims(), 3u);
    EXPECT_EQ(t.total_size(), 24u);
}

TEST(TensorTest, element_access) {
    Tensor<double, 2> t(3, 4);
    t.at(1, 2) = 7.5;
    EXPECT_DOUBLE_EQ(t.at(1, 2), 7.5);
}

TEST(TensorTest, zero_initialized) {
    Tensor<double, 2> t(3, 4);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_DOUBLE_EQ(t.at(i, j), 0.0);
        }
    }
}
