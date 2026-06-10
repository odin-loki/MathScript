#include <gtest/gtest.h>
#include <cmath>
#include <sstream>

#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;
using FMatrix = ColMatrix<float>;
using RMatrix = RowMatrix<double>;

// ---------------------------------------------------------------------------
// Default construction
// ---------------------------------------------------------------------------

TEST(MatrixConstruction, DefaultCtorZeroSize) {
    DMatrix A;
    EXPECT_EQ(A.rows(), 0u);
    EXPECT_EQ(A.cols(), 0u);
    EXPECT_EQ(A.size(), 0u);
    EXPECT_TRUE(A.empty());
}

TEST(MatrixConstruction, SizeCtorShape) {
    DMatrix A(3, 4);
    EXPECT_EQ(A.rows(), 3u);
    EXPECT_EQ(A.cols(), 4u);
    EXPECT_EQ(A.size(), 12u);
    EXPECT_FALSE(A.empty());
}

TEST(MatrixConstruction, SizeCtorDefaultZeroFill) {
    DMatrix A(2, 3);
    // value-initialized elements should be zero (std::vector default)
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_DOUBLE_EQ(A(i, j), 0.0);
}

TEST(MatrixConstruction, InitValueCtor) {
    DMatrix A(2, 2, 7.5);
    EXPECT_DOUBLE_EQ(A(0, 0), 7.5);
    EXPECT_DOUBLE_EQ(A(0, 1), 7.5);
    EXPECT_DOUBLE_EQ(A(1, 0), 7.5);
    EXPECT_DOUBLE_EQ(A(1, 1), 7.5);
}

TEST(MatrixConstruction, InitValueCtorNegative) {
    DMatrix A(3, 3, -1.0);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_DOUBLE_EQ(A(i, j), -1.0);
}

// ---------------------------------------------------------------------------
// Initializer-list construction
// ---------------------------------------------------------------------------

TEST(MatrixConstruction, InitListShapeAndValues) {
    DMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    EXPECT_EQ(A.rows(), 2u);
    EXPECT_EQ(A.cols(), 3u);
    EXPECT_DOUBLE_EQ(A(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(A(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(A(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(A(1, 2), 6.0);
}

TEST(MatrixConstruction, InitList1x1) {
    DMatrix A{{42.0}};
    EXPECT_EQ(A.rows(), 1u);
    EXPECT_EQ(A.cols(), 1u);
    EXPECT_DOUBLE_EQ(A(0, 0), 42.0);
}

TEST(MatrixConstruction, InitListRowMismatchThrows) {
    EXPECT_THROW(
        (DMatrix{{1.0, 2.0}, {3.0}}),
        std::invalid_argument
    );
}

// ---------------------------------------------------------------------------
// Copy construction
// ---------------------------------------------------------------------------

TEST(MatrixConstruction, CopyCtorSameValues) {
    DMatrix A{{1.0, 2.0}, {3.0, 4.0}};
    DMatrix B(A);
    EXPECT_EQ(B.rows(), 2u);
    EXPECT_EQ(B.cols(), 2u);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            EXPECT_DOUBLE_EQ(B(i, j), A(i, j));
}

TEST(MatrixConstruction, CopyCtorIndependent) {
    DMatrix A{{1.0, 2.0}, {3.0, 4.0}};
    DMatrix B(A);
    B(0, 0) = 99.0;
    EXPECT_DOUBLE_EQ(A(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(B(0, 0), 99.0);
}

// ---------------------------------------------------------------------------
// Move construction
// ---------------------------------------------------------------------------

TEST(MatrixConstruction, MoveCtorTransfersData) {
    DMatrix A{{5.0, 6.0}, {7.0, 8.0}};
    DMatrix B(std::move(A));
    EXPECT_EQ(B.rows(), 2u);
    EXPECT_EQ(B.cols(), 2u);
    EXPECT_DOUBLE_EQ(B(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(B(1, 1), 8.0);
}

// ---------------------------------------------------------------------------
// Element access
// ---------------------------------------------------------------------------

TEST(MatrixAccess, WriteAndRead) {
    DMatrix A(3, 3, 0.0);
    A(1, 2) = 3.14;
    EXPECT_DOUBLE_EQ(A(1, 2), 3.14);
}

TEST(MatrixAccess, DataPointerNonNull) {
    DMatrix A(2, 2);
    EXPECT_NE(A.data(), nullptr);
}

TEST(MatrixAccess, ConstDataPointerNonNull) {
    const DMatrix A(2, 2, 1.0);
    EXPECT_NE(A.data(), nullptr);
}

// ---------------------------------------------------------------------------
// Storage-order correctness
// ---------------------------------------------------------------------------

TEST(MatrixStorageOrder, ColMajorLayout) {
    // In ColMajor, column 0 is stored first: data[0]=A(0,0), data[1]=A(1,0)
    DMatrix A{{1.0, 2.0}, {3.0, 4.0}};
    const double* d = A.data();
    EXPECT_DOUBLE_EQ(d[0], 1.0); // A(0,0)
    EXPECT_DOUBLE_EQ(d[1], 3.0); // A(1,0) - col-major
    EXPECT_DOUBLE_EQ(d[2], 2.0); // A(0,1)
    EXPECT_DOUBLE_EQ(d[3], 4.0); // A(1,1)
}

TEST(MatrixStorageOrder, RowMajorLayout) {
    // In RowMajor, row 0 is stored first: data[0]=A(0,0), data[1]=A(0,1)
    RMatrix A{{1.0, 2.0}, {3.0, 4.0}};
    const double* d = A.data();
    EXPECT_DOUBLE_EQ(d[0], 1.0); // A(0,0)
    EXPECT_DOUBLE_EQ(d[1], 2.0); // A(0,1) - row-major
    EXPECT_DOUBLE_EQ(d[2], 3.0); // A(1,0)
    EXPECT_DOUBLE_EQ(d[3], 4.0); // A(1,1)
}

// ---------------------------------------------------------------------------
// Operator+ and Operator-
// ---------------------------------------------------------------------------

TEST(MatrixArithmetic, AdditionShape) {
    DMatrix A(2, 3, 1.0);
    DMatrix B(2, 3, 2.0);
    DMatrix C = A + B;
    EXPECT_EQ(C.rows(), 2u);
    EXPECT_EQ(C.cols(), 3u);
}

TEST(MatrixArithmetic, AdditionValues) {
    DMatrix A{{1.0, 2.0}, {3.0, 4.0}};
    DMatrix B{{5.0, 6.0}, {7.0, 8.0}};
    DMatrix C = A + B;
    EXPECT_DOUBLE_EQ(C(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 8.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 12.0);
}

TEST(MatrixArithmetic, SubtractionValues) {
    DMatrix A{{10.0, 5.0}, {3.0, 8.0}};
    DMatrix B{{1.0,  2.0}, {1.0, 2.0}};
    DMatrix C = A - B;
    EXPECT_DOUBLE_EQ(C(0, 0), 9.0);
    EXPECT_DOUBLE_EQ(C(0, 1), 3.0);
    EXPECT_DOUBLE_EQ(C(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 6.0);
}

TEST(MatrixArithmetic, ScalarMultiplyRight) {
    DMatrix A{{1.0, 2.0}, {3.0, 4.0}};
    DMatrix B = A * 2.0;
    EXPECT_DOUBLE_EQ(B(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(B(1, 1), 8.0);
}

TEST(MatrixArithmetic, ScalarMultiplyLeft) {
    DMatrix A{{1.0, 2.0}, {3.0, 4.0}};
    DMatrix B = 3.0 * A;
    EXPECT_DOUBLE_EQ(B(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(B(1, 1), 12.0);
}

// ---------------------------------------------------------------------------
// Matrix multiplication
// ---------------------------------------------------------------------------

TEST(MatrixArithmetic, MatmulShapeAndValues) {
    DMatrix A{{1.0, 2.0}, {3.0, 4.0}};
    DMatrix B{{5.0, 6.0}, {7.0, 8.0}};
    DMatrix C = A * B;
    EXPECT_EQ(C.rows(), 2u);
    EXPECT_EQ(C.cols(), 2u);
    EXPECT_DOUBLE_EQ(C(0, 0), 19.0);  // 1*5+2*7
    EXPECT_DOUBLE_EQ(C(0, 1), 22.0);  // 1*6+2*8
    EXPECT_DOUBLE_EQ(C(1, 0), 43.0);  // 3*5+4*7
    EXPECT_DOUBLE_EQ(C(1, 1), 50.0);  // 3*6+4*8
}

TEST(MatrixArithmetic, MatmulRectangular) {
    // (2x3) * (3x2) = (2x2)
    DMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    DMatrix B{{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
    DMatrix C = A * B;
    EXPECT_EQ(C.rows(), 2u);
    EXPECT_EQ(C.cols(), 2u);
    EXPECT_DOUBLE_EQ(C(0, 0), 58.0);   // 1*7+2*9+3*11
    EXPECT_DOUBLE_EQ(C(1, 1), 154.0);  // 4*8+5*10+6*12
}

// ---------------------------------------------------------------------------
// Float scalar type
// ---------------------------------------------------------------------------

TEST(MatrixConstruction, FloatMatrix) {
    FMatrix A(2, 2, 1.5f);
    EXPECT_EQ(A.rows(), 2u);
    EXPECT_EQ(A.cols(), 2u);
    EXPECT_FLOAT_EQ(A(0, 0), 1.5f);
}

// ---------------------------------------------------------------------------
// Utility functions: zeros, ones, eye, transpose
// ---------------------------------------------------------------------------

TEST(MatrixUtils, ZerosAllZero) {
    DMatrix Z = zeros<double>(3, 3);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_DOUBLE_EQ(Z(i, j), 0.0);
}

TEST(MatrixUtils, OnesAllOne) {
    DMatrix O = ones<double>(2, 4);
    EXPECT_EQ(O.rows(), 2u);
    EXPECT_EQ(O.cols(), 4u);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 4; ++j)
            EXPECT_DOUBLE_EQ(O(i, j), 1.0);
}

TEST(MatrixUtils, EyeDiagonalIsOne) {
    DMatrix I = eye<double>(4);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            EXPECT_DOUBLE_EQ(I(i, j), i == j ? 1.0 : 0.0);
}

TEST(MatrixUtils, TransposeSwapsDimensions) {
    DMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    auto At = transpose(A);
    EXPECT_EQ(At.rows(), 3u);
    EXPECT_EQ(At.cols(), 2u);
    EXPECT_DOUBLE_EQ(At(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(At(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(At(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(At(2, 1), 6.0);
}

TEST(MatrixUtils, PrintDoesNotCrash) {
    DMatrix A{{1.0, 2.0}, {3.0, 4.0}};
    EXPECT_NO_THROW(A.print());
}
