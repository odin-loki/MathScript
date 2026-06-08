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

TEST(FftExtTest, shift_odd_and_even_lengths) {
    const std::vector<std::complex<double>> even{{0, 0}, {1, 0}, {2, 0}, {3, 0}};
    const auto even_shift = fftshift(even);
    EXPECT_DOUBLE_EQ(even_shift[0].real(), 2.0);
    EXPECT_DOUBLE_EQ(even_shift[1].real(), 3.0);

    const std::vector<std::complex<double>> odd{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};
    const auto odd_shift = fftshift(odd);
    const auto odd_back = ifftshift(odd_shift);
    for (size_t i = 0; i < odd.size(); ++i) {
        EXPECT_DOUBLE_EQ(odd_back[i].real(), odd[i].real());
    }

    const auto even_back = ifftshift(fftshift(even));
    for (size_t i = 0; i < even.size(); ++i) {
        EXPECT_DOUBLE_EQ(even_back[i].real(), even[i].real());
    }
}

TEST(FftExtTest, rfft_irfft_edge_lengths) {
    const std::vector<double> odd{1.0, 2.0, 3.0};
    const auto odd_spec = rfft(odd).value();
    ASSERT_EQ(odd_spec.size(), 3u);
    const auto odd_back = irfft(odd_spec, odd.size()).value();
    for (size_t i = 0; i < odd.size(); ++i) {
        EXPECT_NEAR(odd_back[i], odd[i], 1e-5);
    }

    const std::vector<double> even{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const auto even_spec = rfft(even).value();
    ASSERT_EQ(even_spec.size(), 5u);
    const auto even_back = irfft(even_spec, even.size()).value();
    for (size_t i = 0; i < even.size(); ++i) {
        EXPECT_NEAR(even_back[i], even[i], 1e-5);
    }
}

TEST(FftExtTest, irfft_hermitian_extension) {
    const std::vector<double> x{1.0, 0.5, -0.5, 0.25, 0.75};
    const auto spec = rfft(x).value();
    const auto back = irfft(spec, x.size()).value();
    ASSERT_GE(back.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(back[i], x[i], 1e-5);
    }
}

TEST(FftExtTest, fft2_empty_guard) {
    EXPECT_TRUE(fft2({}).value().empty());
}
