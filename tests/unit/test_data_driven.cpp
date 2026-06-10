#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ms/core/operations.hpp"
#include "ms/fft/fft.hpp"
#include "ms/linalg/linalg.hpp"
#include "ms/special/special.hpp"

using namespace ms;

struct MatrixSizeParam {
    int n;
    double tol;
};

class MatmulDataTest : public ::testing::TestWithParam<MatrixSizeParam> {};

INSTANTIATE_TEST_SUITE_P(MatmulSizes, MatmulDataTest,
    ::testing::Values(
        MatrixSizeParam{2, 1e-12},
        MatrixSizeParam{4, 1e-12},
        MatrixSizeParam{8, 1e-11},
        MatrixSizeParam{16, 1e-10},
        MatrixSizeParam{32, 1e-10}));

TEST_P(MatmulDataTest, identity_product_is_self) {
    const auto [n, tol] = GetParam();
    const auto I = eye<double>(static_cast<size_t>(n));
    const auto A = rand<double>(static_cast<size_t>(n), static_cast<size_t>(n), 42u);
    const auto result = matmul(I, A);
    ASSERT_TRUE(result.has_value());

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR((*result)(static_cast<size_t>(i), static_cast<size_t>(j)),
                        A(static_cast<size_t>(i), static_cast<size_t>(j)),
                        tol);
        }
    }
}

struct SpecialFnParam {
    double x;
    double expected;
    double tol;
    std::string name;
};

class ErfDataTest : public ::testing::TestWithParam<SpecialFnParam> {};

INSTANTIATE_TEST_SUITE_P(ErfValues, ErfDataTest,
    ::testing::Values(
        SpecialFnParam{0.0, 0.0, 1e-15, "erf(0)"},
        SpecialFnParam{0.5, 0.52049987781, 1e-11, "erf(0.5)"},
        SpecialFnParam{1.0, 0.84270079295, 1e-11, "erf(1)"},
        SpecialFnParam{2.0, 0.99532226502, 1e-11, "erf(2)"},
        SpecialFnParam{-1.0, -0.84270079295, 1e-11, "erf(-1)"},
        SpecialFnParam{0.1, 0.11246291601, 1e-11, "erf(0.1)"}));

TEST_P(ErfDataTest, erf_matches_reference) {
    const auto p = GetParam();
    const double value = ms::erf(p.x);
    EXPECT_NEAR(value, p.expected, p.tol) << p.name;
}

class FftRoundtripTest : public ::testing::TestWithParam<int> {};

INSTANTIATE_TEST_SUITE_P(FftSizes, FftRoundtripTest,
    ::testing::Values(4, 8, 16, 32, 64, 128));

TEST_P(FftRoundtripTest, ifft_fft_roundtrip) {
    const int n = GetParam();
    std::vector<double> signal(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        signal[static_cast<size_t>(i)] = std::sin(2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n));
    }

    const auto forward = fft(signal);
    ASSERT_TRUE(forward.has_value());
    const auto backward = ifft(*forward);
    ASSERT_TRUE(backward.has_value());

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR((*backward)[static_cast<size_t>(i)], signal[static_cast<size_t>(i)], 1e-6);
    }
}
