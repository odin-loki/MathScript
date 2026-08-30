// MathScript QR Decomposition Unit Test

#include <gtest/gtest.h>
#include <variant>
#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

TEST(QRTest, 2x2_positive_definite) {
    DMatrix A{{4, 2}, {1, 3}};
    auto [Q, R] = qr(A).value();

    DMatrix prod = Q * R;
    DMatrix diff = A - prod;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_NEAR(diff(i, j), 0.0, 1e-10);
        }
    }
}

TEST(QRTest, 3x3_identity) {
    DMatrix I = eye<double>(3);
    auto [Q, R] = qr(I).value();

    auto QT = transpose(Q);
    auto QQT = QT * Q;

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(QQT(i, j), (i == j) ? 1.0 : 0.0, 1e-10);
        }
    }
}

TEST(QRTest, 3x3_tall_thin) {
    DMatrix A{{1, 2}, {3, 4}, {5, 6}};
    auto [Q, R] = qr(A).value();

    EXPECT_EQ(Q.rows(), 3);
    EXPECT_EQ(Q.cols(), 2);
    EXPECT_EQ(R.rows(), 2);
    EXPECT_EQ(R.cols(), 2);
}

TEST(QRTest, rank_deficient) {
    DMatrix A{{1, 2}, {2, 4}};
    auto [Q, R] = qr(A).value();
    EXPECT_EQ(Q.rows(), 2);
    EXPECT_EQ(R.cols(), 2);
}

TEST(QRTest, empty_is_dimension_mismatch) {
    DMatrix empty(0, 0);
    auto empty_qr = qr(empty);
    ASSERT_FALSE(empty_qr.has_value());
    ASSERT_TRUE(std::holds_alternative<DimensionMismatch>(empty_qr.error()));

    DMatrix no_cols(3, 0);
    auto no_cols_qr = qr(no_cols);
    ASSERT_FALSE(no_cols_qr.has_value());
    ASSERT_TRUE(std::holds_alternative<DimensionMismatch>(no_cols_qr.error()));
    const auto mismatch = std::get<DimensionMismatch>(no_cols_qr.error());
    EXPECT_EQ(mismatch.got_rows, 3u);
    EXPECT_EQ(mismatch.got_cols, 0u);
}

TEST(QRTest, wide_uses_gram_schmidt) {
    DMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    auto factored = qr(A);
    ASSERT_TRUE(factored.has_value());
    const auto [Q, R] = factored.value();

    EXPECT_EQ(Q.rows(), 2u);
    EXPECT_EQ(Q.cols(), 3u);
    EXPECT_EQ(R.rows(), 3u);
    EXPECT_EQ(R.cols(), 3u);

    const DMatrix prod = Q * R;
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(prod(i, j), A(i, j), 1e-8);
        }
    }
}
