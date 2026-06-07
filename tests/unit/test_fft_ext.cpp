#include <gtest/gtest.h>
#include "ms/fft/fft.hpp"
#include <cmath>

using namespace ms;

TEST(FftExtTest, rfft_roundtrip) {
    std::vector<double> x{1, 2, 3, 4};
    auto spec = rfft(x).value();
    auto back = irfft(spec, x.size()).value();
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(back[i], x[i], 1e-6);
    }
}

TEST(FftExtTest, fftshift) {
    std::vector<std::complex<double>> x{{0, 0}, {1, 0}, {2, 0}, {3, 0}};
    auto shifted = fftshift(x);
    EXPECT_DOUBLE_EQ(shifted[0].real(), 2.0);
    EXPECT_DOUBLE_EQ(shifted[2].real(), 0.0);
}

TEST(FftExtTest, dct2_roundtrip) {
    std::vector<double> x{1, 2, 3, 4};
    auto coeffs = dct2(x).value();
    auto back = idct2(coeffs).value();
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(back[i], x[i], 1e-6);
    }
}

TEST(FftExtTest, dst2_basic) {
    std::vector<double> x{1, 0, -1, 0};
    auto y = dst2(x).value();
    ASSERT_EQ(y.size(), x.size());
    EXPECT_GT(std::abs(y[1]), 0.0);
}
