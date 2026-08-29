#include <gtest/gtest.h>
#include "ms/fft/fft.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

TEST(FftExtTest, idst2_roundtrip) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    const auto coeffs = dst2(x).value();
    const auto back = idst2(coeffs).value();
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(back[i], x[i], 1e-6);
    }
}

TEST(FftExtTest, idst2_hand_n1) {
    const std::vector<double> x{3.0};
    const auto s = dst2(x).value();
    const auto back = idst2(s).value();
    EXPECT_NEAR(back[0], 3.0, 1e-10);
}

TEST(FftExtTest, idst2_hand_n2) {
    const std::vector<double> x{1.0, 2.0};
    const auto s = dst2(x).value();
    const auto back = idst2(s).value();
    EXPECT_NEAR(back[0], 1.0, 1e-8);
    EXPECT_NEAR(back[1], 2.0, 1e-8);
}

TEST(FftExtTest, idst2_roundtrip_various_lengths) {
    const std::vector<double> x7{1.0, -0.5, 2.0, 0.0, -1.0, 3.0, 0.25};
    const auto back7 = idst2(dst2(x7).value()).value();
    for (size_t i = 0; i < x7.size(); ++i) {
        EXPECT_NEAR(back7[i], x7[i], 1e-6);
    }

    std::vector<double> x12(12);
    for (size_t i = 0; i < x12.size(); ++i) {
        x12[i] = std::sin(0.3 * static_cast<double>(i));
    }
    const auto back12 = idst2(dst2(x12).value()).value();
    for (size_t i = 0; i < x12.size(); ++i) {
        EXPECT_NEAR(back12[i], x12[i], 1e-6);
    }
}

TEST(FftExtTest, ifft2_roundtrip_square) {
    const std::vector<std::complex<double>> data{{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    const auto spectrum = fft2(data).value();
    const auto back = ifft2(spectrum).value();
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_NEAR(back[i].real(), data[i].real(), 1e-6);
        EXPECT_NEAR(back[i].imag(), data[i].imag(), 1e-6);
    }
}

TEST(FftExtTest, fft2_empty_guard) {
    EXPECT_TRUE(fft2({}).value().empty());
    EXPECT_TRUE(ifft2({}).value().empty());
}

namespace {

std::vector<std::complex<double>> make_rect_grid(size_t rows, size_t cols) {
    std::vector<std::complex<double>> data(rows * cols);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            const double t = static_cast<double>(r * cols + c);
            data[r * cols + c] = {std::sin(0.17 * t), 0.25 * std::cos(0.09 * t)};
        }
    }
    return data;
}

void expect_fft2_rect_roundtrip(size_t rows, size_t cols) {
    const auto data = make_rect_grid(rows, cols);
    const auto spectrum = fft2(data, rows, cols).value();
    ASSERT_EQ(spectrum.size(), rows * cols);
    const auto back = ifft2(spectrum, rows, cols).value();
    ASSERT_EQ(back.size(), rows * cols);
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_NEAR(back[i].real(), data[i].real(), 1e-8) << "rows=" << rows << " cols=" << cols;
        EXPECT_NEAR(back[i].imag(), data[i].imag(), 1e-8) << "rows=" << rows << " cols=" << cols;
    }
}

} // namespace

TEST(FftExtTest, fft2_ifft2_rect_roundtrip_8x16) {
    expect_fft2_rect_roundtrip(8, 16);
}

TEST(FftExtTest, fft2_ifft2_rect_roundtrip_32x64) {
    expect_fft2_rect_roundtrip(32, 64);
}

TEST(FftExtTest, fft2_explicit_dims_mismatch) {
    const std::vector<std::complex<double>> data(128, {1.0, 0.0});
    EXPECT_FALSE(fft2(data, 8, 15).has_value());
    EXPECT_FALSE(ifft2(data, 8, 15).has_value());
}

TEST(FftExtTest, dst2_inverse) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    const auto coeffs = dst2(x).value();
    const auto back = dst2(coeffs).value();
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(back[i], x[i], 1e-6);
    }

    const std::vector<double> zeros(4, 0.0);
    const auto zero_dst = dst2(zeros).value();
    for (double v : zero_dst) {
        EXPECT_DOUBLE_EQ(v, 0.0);
    }
}

TEST(FftExtTest, rfft_irfft_roundtrip) {
    const std::vector<double> signal{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const auto spec = rfft(signal).value();
    const auto back = irfft(spec, signal.size()).value();
    ASSERT_EQ(back.size(), signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        EXPECT_NEAR(back[i], signal[i], 1e-6);
    }
}

TEST(FftExtTest, rfft_irfft_roundtrip_1024) {
    constexpr size_t n = 1024;
    std::vector<double> signal(n);
    for (size_t i = 0; i < n; ++i) {
        signal[i] = std::sin(2.0 * M_PI * 7.0 * static_cast<double>(i) / static_cast<double>(n))
                  + 0.25 * std::cos(2.0 * M_PI * 31.0 * static_cast<double>(i) / static_cast<double>(n));
    }
    const auto spec = rfft(signal).value();
    ASSERT_EQ(spec.size(), n / 2 + 1);
    const auto back = irfft(spec, n).value();
    ASSERT_EQ(back.size(), n);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(back[i], signal[i], 1e-9);
    }
}

TEST(FftExtTest, rfft_irfft_roundtrip_4096) {
    constexpr size_t n = 4096;
    std::vector<double> signal(n);
    for (size_t i = 0; i < n; ++i) {
        signal[i] = std::sin(2.0 * M_PI * 13.0 * static_cast<double>(i) / static_cast<double>(n))
                  + 0.5 * std::cos(2.0 * M_PI * 97.0 * static_cast<double>(i) / static_cast<double>(n));
    }
    const auto spec = rfft(signal).value();
    ASSERT_EQ(spec.size(), n / 2 + 1);
    const auto back = irfft(spec, n).value();
    ASSERT_EQ(back.size(), n);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(back[i], signal[i], 1e-9);
    }
}

TEST(FftExtTest, fftshift_ifftshift_roundtrip) {
    const std::vector<std::complex<double>> x{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}};
    const auto shifted = fftshift(x);
    const auto restored = ifftshift(shifted);
    ASSERT_EQ(restored.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_DOUBLE_EQ(restored[i].real(), x[i].real());
        EXPECT_DOUBLE_EQ(restored[i].imag(), x[i].imag());
    }
}

TEST(FftExtTest, empty_input_dct2) {
    const auto result = dct2({});
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->empty());
}
