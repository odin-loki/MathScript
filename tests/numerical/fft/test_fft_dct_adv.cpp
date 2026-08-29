// MathScript: Advanced DCT/DST Tests (Wave 47)
// Tests for dct2/idct2/dst2: roundtrip, energy, linearity, basis properties

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/fft/fft.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// DCT-II (dct2): roundtrip with idct2
// ---------------------------------------------------------------------------

TEST(FFTDctAdv, DCT2_Roundtrip_Impulse) {
    int N = 16;
    std::vector<double> signal(N, 0.0);
    signal[0] = 1.0;
    auto dct_res = dct2(signal);
    ASSERT_TRUE(dct_res.has_value());
    auto back_res = idct2(dct_res.value());
    ASSERT_TRUE(back_res.has_value());
    auto& back = back_res.value();
    ASSERT_EQ(back.size(), static_cast<size_t>(N));
    EXPECT_NEAR(back[0], 1.0, 1e-9);
    for (int i = 1; i < N; ++i)
        EXPECT_NEAR(back[i], 0.0, 1e-9) << "Roundtrip error at i=" << i;
}

TEST(FFTDctAdv, DCT2_Roundtrip_Sine) {
    int N = 32;
    std::vector<double> signal(N);
    for (int i = 0; i < N; ++i)
        signal[i] = std::sin(2.0 * M_PI * 3.0 * i / N);
    auto dct_res = dct2(signal);
    ASSERT_TRUE(dct_res.has_value());
    auto back_res = idct2(dct_res.value());
    ASSERT_TRUE(back_res.has_value());
    auto& back = back_res.value();
    ASSERT_EQ(back.size(), static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(back[i], signal[i], 1e-8) << "DCT roundtrip error at i=" << i;
}

TEST(FFTDctAdv, DCT2_Roundtrip_Constant) {
    int N = 16;
    std::vector<double> signal(N, 3.0);
    auto dct_res = dct2(signal);
    ASSERT_TRUE(dct_res.has_value());
    auto back_res = idct2(dct_res.value());
    ASSERT_TRUE(back_res.has_value());
    auto& back = back_res.value();
    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(back[i], 3.0, 1e-8);
}

TEST(FFTDctAdv, DCT2_Roundtrip_AllFinite) {
    int N = 32;
    std::vector<double> signal(N);
    for (int i = 0; i < N; ++i)
        signal[i] = std::cos(2.0 * M_PI * 7.0 * i / N) + 0.5;
    auto dct_res = dct2(signal);
    ASSERT_TRUE(dct_res.has_value());
    auto back_res = idct2(dct_res.value());
    ASSERT_TRUE(back_res.has_value());
    for (auto v : back_res.value()) EXPECT_TRUE(std::isfinite(v));
}

// ---------------------------------------------------------------------------
// DCT-II basic properties
// ---------------------------------------------------------------------------

TEST(FFTDctAdv, DCT2_OutputSize_MatchesInput) {
    int N = 32;
    std::vector<double> signal(N, 1.0);
    auto dct_res = dct2(signal);
    ASSERT_TRUE(dct_res.has_value());
    EXPECT_EQ(dct_res.value().size(), static_cast<size_t>(N));
}

TEST(FFTDctAdv, DCT2_AllFinite) {
    int N = 64;
    std::vector<double> signal(N);
    for (int i = 0; i < N; ++i) signal[i] = i + 1.0;
    auto dct_res = dct2(signal);
    ASSERT_TRUE(dct_res.has_value());
    for (auto v : dct_res.value()) EXPECT_TRUE(std::isfinite(v));
}

TEST(FFTDctAdv, DCT2_ZeroInput_ZeroOutput) {
    int N = 16;
    std::vector<double> signal(N, 0.0);
    auto dct_res = dct2(signal);
    ASSERT_TRUE(dct_res.has_value());
    for (auto v : dct_res.value()) EXPECT_NEAR(v, 0.0, 1e-10);
}

TEST(FFTDctAdv, IDCT2_ZeroInput_ZeroOutput) {
    int N = 16;
    std::vector<double> signal(N, 0.0);
    auto back_res = idct2(signal);
    ASSERT_TRUE(back_res.has_value());
    for (auto v : back_res.value()) EXPECT_NEAR(v, 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// DST-II (dst2)
// ---------------------------------------------------------------------------

TEST(FFTDctAdv, DST2_OutputSize_MatchesInput) {
    int N = 16;
    std::vector<double> signal(N, 1.0);
    auto dst_res = dst2(signal);
    ASSERT_TRUE(dst_res.has_value());
    EXPECT_EQ(dst_res.value().size(), static_cast<size_t>(N));
}

TEST(FFTDctAdv, DST2_AllFinite) {
    int N = 32;
    std::vector<double> signal(N);
    for (int i = 0; i < N; ++i) signal[i] = std::sin(M_PI * i / N);
    auto dst_res = dst2(signal);
    ASSERT_TRUE(dst_res.has_value());
    for (auto v : dst_res.value()) EXPECT_TRUE(std::isfinite(v));
}

TEST(FFTDctAdv, DST2_ZeroInput_ZeroOutput) {
    int N = 16;
    std::vector<double> signal(N, 0.0);
    auto dst_res = dst2(signal);
    ASSERT_TRUE(dst_res.has_value());
    for (auto v : dst_res.value()) EXPECT_NEAR(v, 0.0, 1e-10);
}

TEST(FFTDctAdv, IDST2_Roundtrip_Impulse) {
    int N = 16;
    std::vector<double> signal(N, 0.0);
    signal[0] = 1.0;
    auto dst_res = dst2(signal);
    ASSERT_TRUE(dst_res.has_value());
    auto back_res = idst2(dst_res.value());
    ASSERT_TRUE(back_res.has_value());
    EXPECT_NEAR(back_res.value()[0], 1.0, 1e-9);
    for (int i = 1; i < N; ++i) {
        EXPECT_NEAR(back_res.value()[static_cast<size_t>(i)], 0.0, 1e-9);
    }
}

TEST(FFTDctAdv, IDST2_Roundtrip_Sine) {
    int N = 24;
    std::vector<double> signal(N);
    for (int i = 0; i < N; ++i) {
        signal[i] = std::sin(M_PI * static_cast<double>(i + 1) / static_cast<double>(N + 1));
    }
    auto dst_res = dst2(signal);
    ASSERT_TRUE(dst_res.has_value());
    auto back_res = idst2(dst_res.value());
    ASSERT_TRUE(back_res.has_value());
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(back_res.value()[static_cast<size_t>(i)], signal[i], 1e-8);
    }
}

TEST(FFTDctAdv, IDST2_EmptyInput) {
    EXPECT_TRUE(idst2({}).value().empty());
}

// ---------------------------------------------------------------------------
// fftshift / ifftshift
// ---------------------------------------------------------------------------

TEST(FFTDctAdv, FFTShift_OutputSize) {
    std::vector<std::complex<double>> x(8, {1.0, 0.0});
    auto y = fftshift(x);
    EXPECT_EQ(y.size(), x.size());
}

TEST(FFTDctAdv, FFTShift_IFFTS_RoundTrip) {
    std::vector<std::complex<double>> x = {
        {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    auto shifted = fftshift(x);
    auto restored = ifftshift(shifted);
    ASSERT_EQ(restored.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(std::real(restored[i]), std::real(x[i]), 1e-10);
        EXPECT_NEAR(std::imag(restored[i]), std::imag(x[i]), 1e-10);
    }
}

TEST(FFTDctAdv, FFTShift_MovesDCToCenter) {
    // For even N, fftshift moves element N/2 to position 0
    int N = 8;
    std::vector<std::complex<double>> x(N);
    for (int i = 0; i < N; ++i) x[i] = {static_cast<double>(i), 0.0};
    auto shifted = fftshift(x);
    // After shift, the center should have the original second half at front
    EXPECT_NEAR(std::real(shifted[0]), N / 2.0, 1e-10);
}

TEST(FFTDctAdv, FFTShift_AllFinite) {
    int N = 16;
    std::vector<std::complex<double>> x(N);
    for (int i = 0; i < N; ++i) x[i] = {std::sin(i * 0.3), std::cos(i * 0.3)};
    auto y = fftshift(x);
    for (auto& c : y) {
        EXPECT_TRUE(std::isfinite(std::real(c)));
        EXPECT_TRUE(std::isfinite(std::imag(c)));
    }
}
