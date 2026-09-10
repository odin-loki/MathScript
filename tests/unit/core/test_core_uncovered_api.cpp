// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// Small public entry points that no test called: Tensor's accessors, Matrix's
// initializer-list constructor, BigInt's unary minus, the precision() accessors, and the
// float instantiation of expm. Each is one or two lines, so a wrong index or a dropped
// sign in any of them would have gone unnoticed.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "ms/bignum/bignum.hpp"
#include "ms/core/matrix.hpp"
#include "ms/core/tensor.hpp"
#include "ms/linalg/linalg.hpp"

TEST(CoreUncoveredApi, TensorAccessors) {
    ms::Tensor<double, 2> t(3, 4);
    EXPECT_EQ(t.dims(), 2u);
    EXPECT_EQ(t.size(0), 3u);
    EXPECT_EQ(t.size(1), 4u);
    EXPECT_EQ(t.total_size(), 12u);

    // at(i, j) is row-major: at(i, j) and at(i, j+1) must be adjacent.
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            t.at(i, j) = static_cast<double>(i * 10 + j);
        }
    }
    const ms::Tensor<double, 2>& ct = t;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            EXPECT_DOUBLE_EQ(ct.at(i, j), static_cast<double>(i * 10 + j)) << i << "," << j;
        }
    }

    ms::Tensor<float, 2> f(2, 2);
    EXPECT_EQ(f.dims(), 2u);
    f.at(1, 1) = 2.5f;
    const ms::Tensor<float, 2>& cf = f;
    EXPECT_FLOAT_EQ(cf.at(1, 1), 2.5f);

    // reshape preserves the element count.
    const auto r = t.reshape({2, 6});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->total_size(), 12u);
    EXPECT_FALSE(t.reshape({5, 5}).has_value()) << "a reshape must preserve the size";
}

TEST(CoreUncoveredApi, MatrixInitializerListConstructor) {
    const ms::Matrix<double> m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    ASSERT_EQ(m.rows(), 2u);
    ASSERT_EQ(m.cols(), 3u);
    EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(m(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(m(1, 2), 6.0);

    // Ragged rows leave the matrix empty rather than reading past a short row.
    const ms::Matrix<double> ragged{{1.0, 2.0}, {3.0}};
    EXPECT_EQ(ragged.rows() * ragged.cols(), 0u);
}

TEST(CoreUncoveredApi, BigIntUnaryMinusAndPrecisionAccessors) {
    const ms::bignum::BigInt a(12345LL);
    EXPECT_EQ((-a).to_string(), "-12345");
    EXPECT_EQ((-(-a)).to_string(), "12345");
    // Negating zero must not produce "-0".
    const ms::bignum::BigInt z(0LL);
    EXPECT_EQ((-z).to_string(), "0");
    EXPECT_TRUE((-z).is_zero());

    const ms::bignum::APFloat f(1LL, 55);
    EXPECT_EQ(f.precision(), 55);
    const ms::bignum::APComplex c(3LL, 4LL, 45);
    EXPECT_EQ(c.precision(), 45);
    EXPECT_EQ(c.re.precision(), 45);
}

TEST(CoreUncoveredApi, ExpmOnFloatMatrices) {
    // The float instantiation exists but nothing exercised it. exp of a diagonal is the
    // diagonal of the exponentials, and expm(0) is the identity.
    ms::Matrix<float> a(2, 2);
    a(0, 0) = 1.0f;
    a(1, 1) = -1.0f;
    const auto e = ms::expm(a);
    ASSERT_TRUE(e.has_value());
    EXPECT_NEAR((*e)(0, 0), std::exp(1.0f), 1e-5f);
    EXPECT_NEAR((*e)(1, 1), std::exp(-1.0f), 1e-5f);
    EXPECT_NEAR((*e)(0, 1), 0.0f, 1e-6f);

    ms::Matrix<float> zero(3, 3);
    const auto id = ms::expm(zero);
    ASSERT_TRUE(id.has_value());
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR((*id)(i, j), i == j ? 1.0f : 0.0f, 1e-6f);
        }
    }

    // A nilpotent block: expm([[0,1],[0,0]]) == [[1,1],[0,1]] exactly.
    ms::Matrix<float> n(2, 2);
    n(0, 1) = 1.0f;
    const auto en = ms::expm(n);
    ASSERT_TRUE(en.has_value());
    EXPECT_NEAR((*en)(0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR((*en)(0, 1), 1.0f, 1e-6f);
    EXPECT_NEAR((*en)(1, 0), 0.0f, 1e-6f);
    EXPECT_NEAR((*en)(1, 1), 1.0f, 1e-6f);

    ms::Matrix<float> oblong(2, 3);
    EXPECT_FALSE(ms::expm(oblong).has_value());
}
