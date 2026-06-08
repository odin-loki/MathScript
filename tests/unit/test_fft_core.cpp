#include <gtest/gtest.h>
#include <cmath>

#include "ms/fft/fft.hpp"

using namespace ms;

TEST(FftCoreTest, fft_ifft_roundtrip) {
    const std::vector<double> x{1, 2, 3, 4, 5};
    const auto spec = fft(x).value();
    const auto back = ifft(spec).value();
    ASSERT_EQ(back.size(), spec.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(back[i], x[i], 1e-6);
    }
}

TEST(FftCoreTest, odd_length_dft_path) {
    const std::vector<double> x{1, 2, 3};
    const auto spec = dft(std::span<const double>(x)).value();
    ASSERT_EQ(spec.size(), 4u);
    const auto back = ifft(spec).value();
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(back[i], x[i], 1e-5);
    }
}

TEST(FftCoreTest, empty_and_shift) {
    EXPECT_TRUE(fft({}).value().empty());
    EXPECT_TRUE(ifft({}).value().empty());

    const std::vector<std::complex<double>> x{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};
    const auto shifted = ifftshift(fftshift(x));
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_DOUBLE_EQ(shifted[i].real(), x[i].real());
    }
}

TEST(FftCoreTest, fft2_impulse) {
    const std::vector<std::complex<double>> data{{1, 0}, {0, 0}, {0, 0}, {0, 0}};
    const auto out = fft2(data).value();
    ASSERT_EQ(out.size(), 4u);
    for (const auto& v : out) {
        EXPECT_NEAR(v.real(), 1.0, 1e-6);
        EXPECT_NEAR(v.imag(), 0.0, 1e-6);
    }
}

TEST(FftCoreTest, fft2_constant) {
    const std::vector<std::complex<double>> data(4, {1.0, 0.0});
    const auto out = fft2(data).value();
    ASSERT_EQ(out.size(), 4u);
    EXPECT_NEAR(out[0].real(), 4.0, 1e-6);
    for (size_t i = 1; i < out.size(); ++i) {
        EXPECT_NEAR(out[i].real(), 0.0, 1e-6);
        EXPECT_NEAR(out[i].imag(), 0.0, 1e-6);
    }
}

TEST(FftCoreTest, rfft_irfft_roundtrip) {
    const std::vector<double> x{1, 2, 3, 4, 5, 6};
    const auto spec = rfft(x).value();
    const auto back = irfft(spec, x.size()).value();
    ASSERT_GE(back.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(back[i], x[i], 1e-5);
    }
}

TEST(FftCoreTest, dct_and_dst) {
    const std::vector<double> x{1, 2, 3, 4};
    const auto c = dct2(x).value();
    const auto back = idct2(c).value();
    ASSERT_EQ(back.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(back[i], x[i], 1e-4);
    }
    const auto s = dst2(x).value();
    EXPECT_EQ(s.size(), x.size());
}

TEST(FftCoreTest, odd_length_ifft) {
    const std::vector<std::complex<double>> spec{{1, 0}, {2, 0}, {3, 0}};
    const auto back = ifft(spec).value();
    ASSERT_EQ(back.size(), 3u);
    for (size_t i = 0; i < back.size(); ++i) {
        EXPECT_TRUE(std::isfinite(back[i]));
    }
    const std::vector<double> x{1, 2, 3};
    const auto roundtrip = ifft(dft(std::span<const double>(x)).value()).value();
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(roundtrip[i], x[i], 1e-5);
    }
}

TEST(FftCoreTest, empty_shift_and_transforms) {
    const std::vector<std::complex<double>> empty_cx;
    EXPECT_TRUE(fftshift(empty_cx).empty());
    EXPECT_TRUE(ifftshift(empty_cx).empty());

    EXPECT_TRUE(dct2({}).value().empty());
    EXPECT_TRUE(idct2({}).value().empty());
    EXPECT_TRUE(dst2({}).value().empty());

    const std::vector<std::complex<double>> spec{{1, 0}, {0, 0}};
    EXPECT_TRUE(irfft(spec, 0).value().empty());
    EXPECT_TRUE(irfft({}, 8).value().empty());
}
