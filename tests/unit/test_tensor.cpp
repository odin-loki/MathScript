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

TEST(TensorTest, total_size_2d) {
    Tensor<double, 2> t(5, 6);
    EXPECT_EQ(t.total_size(), 30u);
}

TEST(TensorTest, total_size_1x1) {
    Tensor<double, 2> t(1, 1);
    EXPECT_EQ(t.total_size(), 1u);
}

TEST(TensorTest, write_read_multiple_elements) {
    Tensor<double, 2> t(4, 4);
    t.at(0, 0) = 1.0;
    t.at(1, 1) = 2.0;
    t.at(2, 2) = 3.0;
    t.at(3, 3) = 4.0;
    EXPECT_DOUBLE_EQ(t.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(t.at(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(t.at(2, 2), 3.0);
    EXPECT_DOUBLE_EQ(t.at(3, 3), 4.0);
    EXPECT_DOUBLE_EQ(t.at(0, 1), 0.0);  // unset => zero
}

TEST(TensorTest, float_tensor) {
    Tensor<float, 2> t(2, 3);
    t.at(0, 2) = 4.2f;
    EXPECT_FLOAT_EQ(t.at(0, 2), 4.2f);
}

TEST(TensorTest, shape_vector_constructor) {
    const std::vector<size_t> shape = {3u, 5u};
    Tensor<double, 2> t(shape);
    EXPECT_EQ(t.size(0), 3u);
    EXPECT_EQ(t.size(1), 5u);
    EXPECT_EQ(t.total_size(), 15u);
}
