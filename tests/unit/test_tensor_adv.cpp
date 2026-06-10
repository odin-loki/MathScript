#include <gtest/gtest.h>
#include "ms/core/tensor.hpp"

using namespace ms;

// size() returns 0 for a dimension index >= dims()
TEST(TensorAdvTest, SizeOutOfRange_2D) {
    Tensor<double> t(3, 4);
    EXPECT_EQ(t.size(2), 0u);
    EXPECT_EQ(t.size(100), 0u);
}

// 3D tensor: per-dimension size() is correct
TEST(TensorAdvTest, ThreeDims_SizePerDim) {
    Tensor<double> t(2, 5, 7);
    EXPECT_EQ(t.dims(), 3u);
    EXPECT_EQ(t.size(0), 2u);
    EXPECT_EQ(t.size(1), 5u);
    EXPECT_EQ(t.size(2), 7u);
}

// 3D tensor: size() out of range returns 0
TEST(TensorAdvTest, SizeOutOfRange_3D) {
    Tensor<double> t(2, 3, 4);
    EXPECT_EQ(t.size(3), 0u);
}

// 3D tensor: total_size = dim0 * dim1 * dim2
TEST(TensorAdvTest, ThreeDims_TotalSize) {
    Tensor<double> t(3, 4, 5);
    EXPECT_EQ(t.total_size(), 60u);
}

// 1 x N tensor: single row
TEST(TensorAdvTest, SingleRowTensor) {
    Tensor<double> t(1, 8);
    EXPECT_EQ(t.dims(), 2u);
    EXPECT_EQ(t.size(0), 1u);
    EXPECT_EQ(t.size(1), 8u);
    EXPECT_EQ(t.total_size(), 8u);
    t.at(0, 5) = 3.14;
    EXPECT_DOUBLE_EQ(t.at(0, 5), 3.14);
}

// N x 1 tensor: single column
TEST(TensorAdvTest, SingleColumnTensor) {
    Tensor<double> t(6, 1);
    EXPECT_EQ(t.total_size(), 6u);
    t.at(4, 0) = 7.7;
    EXPECT_DOUBLE_EQ(t.at(4, 0), 7.7);
    EXPECT_DOUBLE_EQ(t.at(0, 0), 0.0);
}

// Row-major layout: writing a full row and reading back correctly
TEST(TensorAdvTest, RowMajorLayout) {
    Tensor<double> t(4, 5);
    // Fill row 2 with distinct values
    for (size_t j = 0; j < 5; ++j) {
        t.at(2, j) = static_cast<double>(j * 10);
    }
    // Verify all row-2 values
    for (size_t j = 0; j < 5; ++j) {
        EXPECT_DOUBLE_EQ(t.at(2, j), static_cast<double>(j * 10));
    }
    // Other rows must still be zero
    for (size_t j = 0; j < 5; ++j) {
        EXPECT_DOUBLE_EQ(t.at(0, j), 0.0);
        EXPECT_DOUBLE_EQ(t.at(1, j), 0.0);
    }
}

// Overwriting an element updates the stored value
TEST(TensorAdvTest, ElementOverwrite) {
    Tensor<double> t(3, 3);
    t.at(1, 2) = 5.0;
    EXPECT_DOUBLE_EQ(t.at(1, 2), 5.0);
    t.at(1, 2) = 99.0;
    EXPECT_DOUBLE_EQ(t.at(1, 2), 99.0);
}

// Writes to different elements do not alias each other
TEST(TensorAdvTest, NoAliasing) {
    Tensor<double> t(3, 3);
    t.at(0, 0) = 1.0;
    t.at(0, 1) = 2.0;
    t.at(1, 0) = 3.0;
    t.at(2, 2) = 4.0;
    EXPECT_DOUBLE_EQ(t.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(t.at(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(t.at(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(t.at(2, 2), 4.0);
    // Untouched neighbors remain zero
    EXPECT_DOUBLE_EQ(t.at(0, 2), 0.0);
    EXPECT_DOUBLE_EQ(t.at(1, 1), 0.0);
}

// vector-shape constructor: dims() equals shape.size()
TEST(TensorAdvTest, VectorShapeConstructor_Dims) {
    const std::vector<size_t> shape = {6u, 7u};
    Tensor<double> t(shape);
    EXPECT_EQ(t.dims(), 2u);
    EXPECT_EQ(t.total_size(), 42u);
}

// Non-square tensor: size() per dimension is independent
TEST(TensorAdvTest, NonSquare_SizePerDim) {
    Tensor<double> t(7, 11);
    EXPECT_EQ(t.size(0), 7u);
    EXPECT_EQ(t.size(1), 11u);
    EXPECT_EQ(t.total_size(), 77u);
}

// Const at() can read values written through mutable at()
TEST(TensorAdvTest, ConstAt_MultiplePositions) {
    Tensor<double> t(4, 4);
    t.at(0, 3) = 1.5;
    t.at(3, 0) = 2.5;
    const Tensor<double>& ct = t;
    EXPECT_DOUBLE_EQ(ct.at(0, 3), 1.5);
    EXPECT_DOUBLE_EQ(ct.at(3, 0), 2.5);
    EXPECT_DOUBLE_EQ(ct.at(1, 1), 0.0);
}

// Float 3D tensor: dims and total_size are correct
TEST(TensorAdvTest, FloatTensor3D) {
    Tensor<float> t(2, 3, 4);
    EXPECT_EQ(t.dims(), 3u);
    EXPECT_EQ(t.total_size(), 24u);
    EXPECT_EQ(t.size(0), 2u);
    EXPECT_EQ(t.size(1), 3u);
}

// Large 2D tensor: construction and corner-element access do not throw
TEST(TensorAdvTest, LargeTensor_CornerElements) {
    Tensor<double> t(50, 80);
    EXPECT_EQ(t.total_size(), 4000u);
    t.at(0, 0) = -1.0;
    t.at(49, 79) = 99.0;
    EXPECT_DOUBLE_EQ(t.at(0, 0), -1.0);
    EXPECT_DOUBLE_EQ(t.at(49, 79), 99.0);
    // Spot-check that middle elements are zero
    EXPECT_DOUBLE_EQ(t.at(25, 40), 0.0);
}

// 2D dims() is always 2 for two-arg constructor
TEST(TensorAdvTest, TwoArg_Dims_Always2) {
    Tensor<double> t1(1, 1);
    Tensor<double> t2(100, 1);
    Tensor<double> t3(1, 100);
    EXPECT_EQ(t1.dims(), 2u);
    EXPECT_EQ(t2.dims(), 2u);
    EXPECT_EQ(t3.dims(), 2u);
}
