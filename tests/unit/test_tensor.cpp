#include <gtest/gtest.h>
#include <stdexcept>
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

TEST(TensorTest, reshape_2x3_to_3x2) {
    Tensor<double, 2> t(2, 3);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            t.at(i, j) = static_cast<double>(i * 3 + j);
        }
    }
    const auto r = t.reshape({3u, 2u});
    EXPECT_EQ(r.dims(), 2u);
    EXPECT_EQ(r.size(0), 3u);
    EXPECT_EQ(r.size(1), 2u);
    EXPECT_EQ(r.total_size(), 6u);
    EXPECT_DOUBLE_EQ(r.at(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(r.at(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(r.at(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(r.at(2, 1), 5.0);
}

TEST(TensorTest, reshape_2x3_to_6x1) {
    Tensor<double, 2> t(2, 3);
    t.at(0, 0) = 1.0;
    t.at(0, 1) = 2.0;
    t.at(0, 2) = 3.0;
    t.at(1, 0) = 4.0;
    t.at(1, 1) = 5.0;
    t.at(1, 2) = 6.0;
    const auto r = t.reshape({6u, 1u});
    EXPECT_EQ(r.size(0), 6u);
    EXPECT_EQ(r.size(1), 1u);
    EXPECT_DOUBLE_EQ(r.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(r.at(3, 0), 4.0);
    EXPECT_DOUBLE_EQ(r.at(5, 0), 6.0);
}

TEST(TensorTest, reshape_2x3_to_1x6) {
    Tensor<double, 2> t(2, 3);
    t.at(1, 2) = 9.0;
    const auto r = t.reshape({1u, 6u});
    EXPECT_EQ(r.size(0), 1u);
    EXPECT_EQ(r.size(1), 6u);
    EXPECT_DOUBLE_EQ(r.at(0, 5), 9.0);
}

TEST(TensorTest, reshape_preserves_row_major_order) {
    Tensor<double, 2> t(2, 3);
    std::vector<double> expected;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            const double v = static_cast<double>(i * 3 + j);
            t.at(i, j) = v;
            expected.push_back(v);
        }
    }
    const auto r = t.reshape({3u, 2u});
    for (size_t k = 0; k < 6; ++k) {
        EXPECT_DOUBLE_EQ(r.at(k / 2, k % 2), expected[k]);
    }
}

TEST(TensorTest, reshape_same_shape_copies_data) {
    Tensor<double, 2> t(2, 3);
    t.at(0, 1) = 42.0;
    const auto r = t.reshape({2u, 3u});
    EXPECT_EQ(r.size(0), 2u);
    EXPECT_EQ(r.size(1), 3u);
    EXPECT_DOUBLE_EQ(r.at(0, 1), 42.0);
}

TEST(TensorTest, reshape_invalid_shape_too_small) {
    Tensor<double, 2> t(2, 3);
    EXPECT_THROW(t.reshape({2u, 2u}), std::invalid_argument);
}

TEST(TensorTest, reshape_invalid_shape_too_large) {
    Tensor<double, 2> t(2, 3);
    EXPECT_THROW(t.reshape({3u, 3u}), std::invalid_argument);
}

TEST(TensorTest, reshape_invalid_shape_wrong_dims) {
    Tensor<double, 2> t(2, 3);
    EXPECT_THROW(t.reshape({2u, 2u, 2u}), std::invalid_argument);
}
