// MathScript QR Decomposition Unit Test

#include <gtest/gtest.h>
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
