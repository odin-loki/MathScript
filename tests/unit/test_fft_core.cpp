#include <gtest/gtest.h>
#include <cmath>

#include "ms/fft/fft.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
    EXPECT_TRUE(idst2({}).value().empty());

    const std::vector<std::complex<double>> spec{{1, 0}, {0, 0}};
    EXPECT_TRUE(irfft(spec, 0).value().empty());
    EXPECT_TRUE(irfft({}, 8).value().empty());
}

TEST(FftCoreTest, goertzel_matches_full_fft) {
    // Compare goertzel() against the corresponding bin of a full fft() for
    // several signal lengths and target frequencies.
    const std::vector<size_t> lengths{8, 16, 32, 64};
    const double fs = 64.0;
    for (size_t n : lengths) {
        std::vector<double> x(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = std::sin(0.37 * static_cast<double>(i)) + 0.5 * std::cos(1.1 * static_cast<double>(i));
        }
        const auto spectrum = fft(x).value();
        ASSERT_EQ(spectrum.size(), n);

        for (double f : {0.0, fs / 8.0, fs / 4.0, 3.0 * fs / 8.0, fs / 2.0}) {
            const size_t k = static_cast<size_t>(std::llround(f / fs * static_cast<double>(n)));
            ASSERT_LT(k, spectrum.size());
            const auto g = goertzel(std::span<const double>(x), f, fs);
            EXPECT_NEAR(g.real(), spectrum[k].real(), 1e-9);
            EXPECT_NEAR(g.imag(), spectrum[k].imag(), 1e-9);
        }
    }
}

TEST(FftCoreTest, goertzel_pure_sinusoid_peak) {
    // A pure sinusoid at frequency f0 should produce a large-magnitude
    // response at f0 and a near-zero response far from f0.
    const size_t n = 128;
    const double fs = 128.0;
    const double f0 = 20.0;
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::sin(2.0 * M_PI * f0 * static_cast<double>(i) / fs);
    }

    const auto at_peak = goertzel(std::span<const double>(x), f0, fs);
    const auto off_peak = goertzel(std::span<const double>(x), 50.0, fs);

    EXPECT_GT(std::abs(at_peak), 0.4 * static_cast<double>(n));
    EXPECT_LT(std::abs(off_peak), 1e-6 * static_cast<double>(n) + 1e-6);
    EXPECT_GT(std::abs(at_peak), 1000.0 * std::abs(off_peak) + 1.0);
}

TEST(FftCoreTest, goertzel_dc_signal) {
    // A constant signal has all of its energy in the DC bin (f = 0): the
    // response there should equal the sum of samples, and non-zero
    // frequency bins should be near zero.
    const size_t n = 32;
    const double fs = 32.0;
    const double value = 3.5;
    const std::vector<double> x(n, value);

    const auto dc = goertzel(std::span<const double>(x), 0.0, fs);
    EXPECT_NEAR(dc.real(), value * static_cast<double>(n), 1e-9);
    EXPECT_NEAR(dc.imag(), 0.0, 1e-9);

    for (double f : {fs / 8.0, fs / 4.0, 3.0 * fs / 8.0}) {
        const auto bin = goertzel(std::span<const double>(x), f, fs);
        EXPECT_NEAR(std::abs(bin), 0.0, 1e-9);
    }
}

TEST(FftCoreTest, goertzel_edge_cases) {
    const std::vector<double> x{1, 2, 3, 4};

    // Empty input.
    EXPECT_EQ(goertzel(std::span<const double>(std::vector<double>{}), 1.0, 8.0), std::complex<double>(0.0, 0.0));

    // Invalid sample rate.
    EXPECT_EQ(goertzel(std::span<const double>(x), 1.0, 0.0), std::complex<double>(0.0, 0.0));
    EXPECT_EQ(goertzel(std::span<const double>(x), 1.0, -8.0), std::complex<double>(0.0, 0.0));

    // Frequency outside [0, fs/2].
    EXPECT_EQ(goertzel(std::span<const double>(x), -1.0, 8.0), std::complex<double>(0.0, 0.0));
    EXPECT_EQ(goertzel(std::span<const double>(x), 5.0, 8.0), std::complex<double>(0.0, 0.0));

    // Boundary frequencies (0 and Nyquist) should not be rejected and
    // should produce finite results.
    const auto dc = goertzel(std::span<const double>(x), 0.0, 8.0);
    EXPECT_TRUE(std::isfinite(dc.real()) && std::isfinite(dc.imag()));
    const auto nyquist = goertzel(std::span<const double>(x), 4.0, 8.0);
    EXPECT_TRUE(std::isfinite(nyquist.real()) && std::isfinite(nyquist.imag()));
}
