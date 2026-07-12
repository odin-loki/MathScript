#include <gtest/gtest.h>
#include <cmath>

#include "ms/fft/fft.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

TEST(FftFreqTest, fftfreq_even_n_matches_numpy) {
    const auto freqs = fftfreq(8, 1.0);
    const std::vector<double> expected{0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125};
    ASSERT_EQ(freqs.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(freqs[i], expected[i], 1e-12);
    }
}

TEST(FftFreqTest, fftfreq_odd_n_matches_numpy) {
    const auto freqs = fftfreq(5, 1.0);
    const std::vector<double> expected{0.0, 0.2, 0.4, -0.4, -0.2};
    ASSERT_EQ(freqs.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(freqs[i], expected[i], 1e-12);
    }
}

TEST(FftFreqTest, fftfreq_nonunit_spacing_scales_by_inverse_d) {
    const double d = 0.5;
    const auto freqs_unit = fftfreq(8, 1.0);
    const auto freqs_scaled = fftfreq(8, d);
    ASSERT_EQ(freqs_unit.size(), freqs_scaled.size());
    for (size_t i = 0; i < freqs_unit.size(); ++i) {
        EXPECT_NEAR(freqs_scaled[i], freqs_unit[i] / d, 1e-12);
    }
}

TEST(FftFreqTest, rfftfreq_even_n_matches_numpy) {
    const auto freqs = rfftfreq(8, 1.0);
    const std::vector<double> expected{0.0, 0.125, 0.25, 0.375, 0.5};
    ASSERT_EQ(freqs.size(), 5u);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(freqs[i], expected[i], 1e-12);
    }
}

TEST(FftFreqTest, rfftfreq_odd_n_matches_numpy) {
    const auto freqs = rfftfreq(5, 1.0);
    const std::vector<double> expected{0.0, 0.2, 0.4};
    ASSERT_EQ(freqs.size(), 3u);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(freqs[i], expected[i], 1e-12);
    }
}

TEST(FftFreqTest, fftfreq_and_rfftfreq_lengths_across_n_values) {
    for (size_t n = 0; n <= 12; ++n) {
        EXPECT_EQ(fftfreq(n).size(), n);
        EXPECT_EQ(rfftfreq(n).size(), n / 2 + (n == 0 ? 0 : 1));
    }
}

TEST(FftFreqTest, zero_length_returns_empty_without_crashing) {
    EXPECT_TRUE(fftfreq(0).empty());
    EXPECT_TRUE(fftfreq(0, 2.0).empty());
    EXPECT_TRUE(rfftfreq(0).empty());
    EXPECT_TRUE(rfftfreq(0, 2.0).empty());
}

TEST(FftFreqTest, fft_peak_bin_matches_fftfreq_known_sinusoid) {
    const size_t n = 64;
    const double d = 1.0 / 64.0; // sample spacing so that fs = 1/d = 64 Hz
    const double target_freq = 8.0; // cycles per unit time, an exact bin (n*d*f = 8)
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::sin(2.0 * M_PI * target_freq * static_cast<double>(i) * d);
    }

    const auto spectrum = fft(x).value();
    const auto freqs = fftfreq(n, d);
    ASSERT_EQ(spectrum.size(), n);
    ASSERT_EQ(freqs.size(), n);

    size_t peak_idx = 0;
    double peak_mag = 0.0;
    for (size_t i = 1; i < n / 2; ++i) {
        const double mag = std::abs(spectrum[i]);
        if (mag > peak_mag) {
            peak_mag = mag;
            peak_idx = i;
        }
    }
    EXPECT_NEAR(freqs[peak_idx], target_freq, 1e-9);
}

TEST(FftFreqTest, rfft_peak_bin_matches_rfftfreq_known_sinusoid) {
    const size_t n = 64;
    const double d = 1.0 / 64.0;
    const double target_freq = 8.0;
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::cos(2.0 * M_PI * target_freq * static_cast<double>(i) * d);
    }

    const auto spectrum = rfft(x).value();
    const auto freqs = rfftfreq(n, d);
    ASSERT_EQ(spectrum.size(), freqs.size());

    size_t peak_idx = 0;
    double peak_mag = 0.0;
    for (size_t i = 1; i < spectrum.size(); ++i) {
        const double mag = std::abs(spectrum[i]);
        if (mag > peak_mag) {
            peak_mag = mag;
            peak_idx = i;
        }
    }
    EXPECT_NEAR(freqs[peak_idx], target_freq, 1e-9);
}
