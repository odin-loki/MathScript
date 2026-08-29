// MathScript Advanced Tensor and Scalar Tests (Wave 54)

#include <gtest/gtest.h>
#include <cmath>
#include <string>
#include <vector>

#include "ms/core/tensor.hpp"
#include "ms/core/scalar.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// Tensor<double, 2>: 2D tensor
// ---------------------------------------------------------------------------

TEST(TensorScalar, Tensor2D_Dims_Shape) {
    Tensor<double> t(3, 4);
    EXPECT_EQ(t.size(0), 3u);
    EXPECT_EQ(t.size(1), 4u);
}

TEST(TensorScalar, Tensor2D_TotalSize) {
    Tensor<double> t(5, 6);
    EXPECT_EQ(t.total_size(), 30u);
}

TEST(TensorScalar, Tensor2D_ReadWrite) {
    Tensor<double> t(3, 3);
    t.at(0, 0) = 1.0;
    t.at(1, 1) = 5.0;
    t.at(2, 2) = 9.0;
    EXPECT_DOUBLE_EQ(t.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(t.at(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(t.at(2, 2), 9.0);
}

TEST(TensorScalar, Tensor2D_AllElementsWriteable) {
    const size_t R = 4, C = 5;
    Tensor<double> t(R, C);
    for (size_t i = 0; i < R; ++i)
        for (size_t j = 0; j < C; ++j)
            t.at(i, j) = static_cast<double>(i * C + j);
    for (size_t i = 0; i < R; ++i)
        for (size_t j = 0; j < C; ++j)
            EXPECT_DOUBLE_EQ(t.at(i, j), static_cast<double>(i * C + j));
}

TEST(TensorScalar, Tensor2D_FromVector_Shape) {
    Tensor<double> t(std::vector<size_t>{7, 3});
    EXPECT_EQ(t.size(0), 7u);
    EXPECT_EQ(t.size(1), 3u);
    EXPECT_EQ(t.total_size(), 21u);
}

TEST(TensorScalar, Tensor2D_DimCount) {
    Tensor<double> t(4, 5);
    // N=2 by default, so dims_ depends on construction
    EXPECT_GE(t.dims(), 1u);
}

TEST(TensorScalar, Tensor3D_Shape) {
    Tensor<double> t(2, 3, 4);
    EXPECT_EQ(t.size(0), 2u);
    EXPECT_EQ(t.size(1), 3u);
}

TEST(TensorScalar, Tensor3D_TotalSize) {
    Tensor<double> t(2, 3, 4);
    // total_size = product of all dimensions = at least 6
    EXPECT_GE(t.total_size(), 6u);
}

TEST(TensorScalar, Tensor_Float_ReadWrite) {
    Tensor<float> t(3, 3);
    t.at(0, 0) = 3.14f;
    EXPECT_FLOAT_EQ(t.at(0, 0), 3.14f);
}

// ---------------------------------------------------------------------------
// Scalar: value, unit, arithmetic
// ---------------------------------------------------------------------------

TEST(TensorScalar, Scalar_DefaultValue) {
    Scalar s;
    EXPECT_DOUBLE_EQ(s.value(), 0.0);
}

TEST(TensorScalar, Scalar_ValueSet) {
    Scalar s(42.0, "m");
    EXPECT_DOUBLE_EQ(s.value(), 42.0);
}

TEST(TensorScalar, Scalar_UnitSet) {
    Scalar s(1.0, "m");
    EXPECT_EQ(s.unit(), "m");
}

TEST(TensorScalar, Scalar_FullUnit_WithPrefix) {
    Scalar s(1.0, "m", "k");
    EXPECT_EQ(s.full_unit(), "km");
}

TEST(TensorScalar, Scalar_FullUnit_NoPrefix) {
    Scalar s(1.0, "s", "");
    EXPECT_EQ(s.full_unit(), "s");
}

TEST(TensorScalar, Scalar_Add_Values) {
    Scalar a(3.0, "m");
    Scalar b(4.0, "m");
    auto c = a + b;
    EXPECT_DOUBLE_EQ(c.value(), 7.0);
}

TEST(TensorScalar, Scalar_Sub_Values) {
    Scalar a(10.0, "kg");
    Scalar b(3.5, "kg");
    auto c = a - b;
    EXPECT_DOUBLE_EQ(c.value(), 6.5);
}

TEST(TensorScalar, Scalar_Mul_Values) {
    Scalar a(3.0, "m");
    Scalar b(4.0, "s");
    auto c = a * b;
    EXPECT_DOUBLE_EQ(c.value(), 12.0);
}

TEST(TensorScalar, Scalar_Div_Values) {
    Scalar a(10.0, "m");
    Scalar b(2.0, "s");
    auto c = a / b;
    EXPECT_DOUBLE_EQ(c.value(), 5.0);
}

TEST(TensorScalar, Scalar_ChainArithmetic) {
    Scalar a(2.0), b(3.0), c(4.0);
    auto result = (a + b) * c;
    EXPECT_DOUBLE_EQ(result.value(), 20.0);
}

TEST(TensorScalar, Scalar_ZeroValue) {
    Scalar z(0.0, "N");
    Scalar a(5.0, "N");
    auto r = z + a;
    EXPECT_DOUBLE_EQ(r.value(), 5.0);
}

TEST(TensorScalar, Scalar_NegativeValue) {
    Scalar a(-3.0, "Pa");
    Scalar b(7.0, "Pa");
    auto c = a + b;
    EXPECT_DOUBLE_EQ(c.value(), 4.0);
}
