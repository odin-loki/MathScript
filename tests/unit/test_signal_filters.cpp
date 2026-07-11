#include <gtest/gtest.h>

#include "ms/signal/signal.hpp"
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

TEST(SignalFilterTest, butterworth_and_lowpass) {
    const std::vector<double> x{0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0};
    const auto bw = butterworth(x, 0.25, 1.0);
    ASSERT_EQ(bw.size(), x.size());
    const auto lp = lowpass(x, 0.25, 1.0);
    ASSERT_EQ(lp.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(lp[i], bw[i], 1e-12);
    }
}

TEST(SignalFilterTest, highpass_and_bandpass) {
    const std::vector<double> x{0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0};
    const auto hp = highpass(x, 0.1, 1.0);
    ASSERT_EQ(hp.size(), x.size());
    const auto bp = bandpass(x, 0.05, 0.4, 1.0);
    ASSERT_EQ(bp.size(), x.size());
}

TEST(SignalFilterTest, empty_input) {
    const std::vector<double> empty;
    EXPECT_TRUE(butterworth(empty, 0.2, 1.0).empty());
    EXPECT_TRUE(highpass(empty, 0.2, 1.0).empty());
}

TEST(SignalFilterTest, window_functions) {
    const auto hann = hanning(8);
    const auto black = blackman(8);
    const auto parz = parzen(8);
    const auto tri = triangular(8);
    ASSERT_EQ(hann.size(), 8u);
    ASSERT_EQ(black.size(), 8u);
    ASSERT_EQ(parz.size(), 8u);
    ASSERT_EQ(tri.size(), 8u);
    EXPECT_NEAR(hann[0], 0.0, 1e-12);
    EXPECT_NEAR(hann[7], 0.0, 1e-12);
    EXPECT_GT(*std::max_element(black.begin(), black.end()), 0.9);
    EXPECT_NEAR(tri[0], 1.0 / 5.0, 1e-12);
    EXPECT_EQ(hanning(0).size(), 0u);
    EXPECT_DOUBLE_EQ(hanning(1)[0], 1.0);
}

TEST(SignalResampleTest, upsample_basic) {
    const std::vector<double> x{1.0, 2.0, 3.0};
    const auto up = upsample(x, 3);
    const std::vector<double> expected{1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0};
    ASSERT_EQ(up.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(up[i], expected[i]);
    }
}

TEST(SignalResampleTest, upsample_invalid_n_or_empty) {
    const std::vector<double> x{1.0, 2.0, 3.0};
    EXPECT_TRUE(upsample(x, 0).empty());
    EXPECT_TRUE(upsample(x, -2).empty());
    EXPECT_TRUE(upsample(std::vector<double>{}, 3).empty());
}

TEST(SignalResampleTest, downsample_basic) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const auto down = downsample(x, 2);
    const std::vector<double> expected{1.0, 3.0, 5.0};
    ASSERT_EQ(down.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(down[i], expected[i]);
    }
}

TEST(SignalResampleTest, downsample_invalid_n_or_empty) {
    const std::vector<double> x{1.0, 2.0, 3.0};
    EXPECT_TRUE(downsample(x, 0).empty());
    EXPECT_TRUE(downsample(x, -1).empty());
    EXPECT_TRUE(downsample(std::vector<double>{}, 2).empty());
}

TEST(SignalResampleTest, upsample_downsample_round_trip) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    for (int n = 1; n <= 4; ++n) {
        const auto round_tripped = downsample(upsample(x, n), n);
        ASSERT_EQ(round_tripped.size(), x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            EXPECT_DOUBLE_EQ(round_tripped[i], x[i]);
        }
    }
}

TEST(SignalResampleTest, decimate_invalid_q_or_empty) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    EXPECT_TRUE(decimate(x, 0).empty());
    EXPECT_TRUE(decimate(x, -3).empty());
    EXPECT_TRUE(decimate(std::vector<double>{}, 2).empty());
}

TEST(SignalResampleTest, decimate_basic_length_and_low_freq_preserved) {
    const size_t N = 200;
    const double f_low = 0.01;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = std::sin(2.0 * M_PI * f_low * static_cast<double>(i));
    }
    const int q = 4;
    const auto dec = decimate(x, q);
    ASSERT_EQ(dec.size(), x.size() / static_cast<size_t>(q));
    // Interior samples should closely track the (unaliased, slowly-varying) low-frequency sine.
    for (size_t i = 5; i + 5 < dec.size(); ++i) {
        const double expected = std::sin(2.0 * M_PI * f_low * static_cast<double>(i * static_cast<size_t>(q)));
        EXPECT_NEAR(dec[i], expected, 0.1);
    }
}

TEST(SignalResampleTest, decimate_anti_aliasing_beats_naive_downsample) {
    const size_t N = 400;
    const int q = 4;
    const double f_low = 0.01;
    const double f_high = 0.4; // well above the new Nyquist (1/(2*q) = 0.125), aliases if not filtered

    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = std::sin(2.0 * M_PI * f_low * static_cast<double>(i)) +
               std::sin(2.0 * M_PI * f_high * static_cast<double>(i));
    }

    const auto dec = decimate(x, q);
    const auto naive = downsample(x, q);
    ASSERT_EQ(dec.size(), naive.size());

    double err_decimate = 0.0;
    double err_naive = 0.0;
    for (size_t i = 0; i < dec.size(); ++i) {
        const double reference = std::sin(2.0 * M_PI * f_low * static_cast<double>(i * static_cast<size_t>(q)));
        err_decimate += (dec[i] - reference) * (dec[i] - reference);
        err_naive += (naive[i] - reference) * (naive[i] - reference);
    }
    err_decimate = std::sqrt(err_decimate / static_cast<double>(dec.size()));
    err_naive = std::sqrt(err_naive / static_cast<double>(naive.size()));

    EXPECT_LT(err_decimate, 0.3);
    EXPECT_GT(err_naive, 0.5);
    EXPECT_LT(err_decimate, err_naive * 0.5);
}

TEST(SignalResampleTest, interpolate_invalid_p_or_empty) {
    const std::vector<double> x{1.0, 2.0, 3.0};
    EXPECT_TRUE(interpolate(x, 0).empty());
    EXPECT_TRUE(interpolate(x, -1).empty());
    EXPECT_TRUE(interpolate(std::vector<double>{}, 3).empty());
}

TEST(SignalResampleTest, interpolate_length) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    const auto out = interpolate(x, 5);
    EXPECT_EQ(out.size(), x.size() * 5);
}

TEST(SignalResampleTest, interpolate_tracks_low_freq_sine) {
    const size_t N = 60;
    const int p = 4;
    const double f = 0.03; // slowly-varying relative to both the original and upsampled rate

    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = std::sin(2.0 * M_PI * f * static_cast<double>(i));
    }

    const auto up = interpolate(x, p);
    ASSERT_EQ(up.size(), N * static_cast<size_t>(p));

    // Interior interpolated samples should track sin(2*pi*f*j/p), the continuous-time value
    // implied by treating x as unit-sample-rate; avoid the boundary region where FFT-based
    // filtering has transient edge effects.
    const size_t margin = 5 * static_cast<size_t>(p);
    for (size_t j = margin; j + margin < up.size(); ++j) {
        const double expected = std::sin(2.0 * M_PI * f * static_cast<double>(j) / static_cast<double>(p));
        EXPECT_NEAR(up[j], expected, 0.15);
    }
}

TEST(SignalResampleTest, interpolate_preserves_original_samples_approximately) {
    const size_t N = 40;
    const int p = 4;
    const double f = 0.02;

    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = std::sin(2.0 * M_PI * f * static_cast<double>(i));
    }

    const auto up = interpolate(x, p);
    for (size_t i = 5; i + 5 < N; ++i) {
        EXPECT_NEAR(up[i * static_cast<size_t>(p)], x[i], 0.1);
    }
}

TEST(SignalResampleTest, resample_invalid_args_or_empty) {
    const std::vector<double> x{1.0, 2.0, 3.0};
    EXPECT_TRUE(resample(x, 0, 2).empty());
    EXPECT_TRUE(resample(x, 2, 0).empty());
    EXPECT_TRUE(resample(x, -1, 2).empty());
    EXPECT_TRUE(resample(std::vector<double>{}, 2, 3).empty());
}

TEST(SignalResampleTest, resample_length_sanity) {
    const size_t N = 100;
    const double f = 0.02;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = std::sin(2.0 * M_PI * f * static_cast<double>(i));
    }

    const int p = 3;
    const int q = 2;
    const auto out = resample(x, p, q);
    const double expected_len = static_cast<double>(N) * static_cast<double>(p) / static_cast<double>(q);
    EXPECT_NEAR(static_cast<double>(out.size()), expected_len, expected_len * 0.05 + 2.0);
    EXPECT_FALSE(out.empty());
}
