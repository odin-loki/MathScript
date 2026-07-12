#include <gtest/gtest.h>

#include "ms/signal/signal.hpp"
#include <algorithm>
#include <cmath>
#include <random>

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

// ---------------------------------------------------------------------------
// filter() / filtfilt(): generic direct-form IIR/FIR application.
// ---------------------------------------------------------------------------

TEST(SignalFilterApplyTest, filter_fir_identity) {
    const std::vector<double> x{1.0, -2.0, 3.5, 0.0, -7.25};
    const auto y = filter({1.0}, {1.0}, x);
    ASSERT_EQ(y.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_DOUBLE_EQ(y[i], x[i]);
    }
}

TEST(SignalFilterApplyTest, filter_fir_moving_average_step_hand_computed) {
    // b = {1/3,1/3,1/3}, a = {1}; x is a step from 0 to 1 at n=3.
    const std::vector<double> b{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
    const std::vector<double> a{1.0};
    const std::vector<double> x{0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const auto y = filter(b, a, x);
    ASSERT_EQ(y.size(), x.size());
    const std::vector<double> expected{0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y[i], expected[i], 1e-12);
    }
}

TEST(SignalFilterApplyTest, filter_iir_exponential_moving_average_hand_computed) {
    // b = {alpha}, a = {1, -(1-alpha)} => y[n] = alpha*x[n] + (1-alpha)*y[n-1].
    const double alpha = 0.5;
    const std::vector<double> b{alpha};
    const std::vector<double> a{1.0, -(1.0 - alpha)};
    const std::vector<double> x{1.0, 1.0, 1.0, 1.0, 1.0};
    const auto y = filter(b, a, x);
    ASSERT_EQ(y.size(), x.size());
    const std::vector<double> expected{0.5, 0.75, 0.875, 0.9375, 0.96875};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y[i], expected[i], 1e-12);
    }
}

TEST(SignalFilterApplyTest, filter_general_iir_hand_computed) {
    // y[n] = x[n] - x[n-1] + 0.5*y[n-1], i.e. b = {1,-1}, a = {1,-0.5}.
    const std::vector<double> b{1.0, -1.0};
    const std::vector<double> a{1.0, -0.5};
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    const auto y = filter(b, a, x);
    ASSERT_EQ(y.size(), x.size());
    const std::vector<double> expected{1.0, 1.5, 1.75, 1.875, 1.9375};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y[i], expected[i], 1e-12);
    }
}

TEST(SignalFilterApplyTest, filter_normalizes_when_a0_not_one) {
    // Same recurrence as filter_general_iir_hand_computed, but scaled by 2:
    // b = {2,-2}, a = {2,-1} should produce identical output once normalized by a[0]=2.
    const std::vector<double> b{2.0, -2.0};
    const std::vector<double> a{2.0, -1.0};
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    const auto y = filter(b, a, x);
    const std::vector<double> expected{1.0, 1.5, 1.75, 1.875, 1.9375};
    ASSERT_EQ(y.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y[i], expected[i], 1e-12);
    }
}

TEST(SignalFilterApplyTest, filter_empty_input_or_b) {
    const std::vector<double> x{1.0, 2.0, 3.0};
    EXPECT_TRUE(filter({1.0}, {1.0}, {}).empty());
    EXPECT_TRUE(filter({}, {1.0}, x).empty());
}

TEST(SignalFilterApplyTest, filtfilt_preserves_length) {
    const std::vector<double> b{0.25, 0.5, 0.25};
    const std::vector<double> a{1.0};
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const auto y = filtfilt(b, a, x);
    EXPECT_EQ(y.size(), x.size());
}

TEST(SignalFilterApplyTest, filtfilt_empty_input_or_b) {
    EXPECT_TRUE(filtfilt({1.0}, {1.0}, {}).empty());
    EXPECT_TRUE(filtfilt({}, {1.0}, {1.0, 2.0}).empty());
}

TEST(SignalFilterApplyTest, filtfilt_is_zero_phase_vs_causal_filter_delay) {
    // A causal FIR lowpass filter has a constant group delay of (n_taps-1)/2 samples;
    // filtfilt should remove that delay entirely.
    const int n_taps = 41;
    const int delay = (n_taps - 1) / 2;
    const auto taps = firwin(n_taps, 0.3, FirWindow::Hamming);
    ASSERT_EQ(taps.size(), static_cast<size_t>(n_taps));

    const size_t N = 300;
    const double f = 0.02; // well within the 0.15-cycles/sample passband
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = std::sin(2.0 * M_PI * f * static_cast<double>(i));
    }

    const auto causal = filter(taps, {1.0}, x);
    const auto zero_phase = filtfilt(taps, {1.0}, x);
    ASSERT_EQ(causal.size(), x.size());
    ASSERT_EQ(zero_phase.size(), x.size());

    const size_t margin = 60;
    double sq_err_undelayed = 0.0;
    double sq_err_delayed = 0.0;
    double sq_err_zero_phase = 0.0;
    size_t count = 0;
    for (size_t i = margin; i + margin < N; ++i) {
        const double ref = x[i];
        sq_err_undelayed += (causal[i] - ref) * (causal[i] - ref);
        sq_err_delayed += (causal[i + static_cast<size_t>(delay)] - ref) * (causal[i + static_cast<size_t>(delay)] - ref);
        sq_err_zero_phase += (zero_phase[i] - ref) * (zero_phase[i] - ref);
        ++count;
    }
    const double rmse_undelayed = std::sqrt(sq_err_undelayed / static_cast<double>(count));
    const double rmse_delayed = std::sqrt(sq_err_delayed / static_cast<double>(count));
    const double rmse_zero_phase = std::sqrt(sq_err_zero_phase / static_cast<double>(count));

    // The causal filter output, compared directly (no shift) to the input, is way off due to
    // the group delay; shifting it by the known delay brings it back in line.
    EXPECT_GT(rmse_undelayed, 0.3);
    EXPECT_LT(rmse_delayed, 0.05);
    // filtfilt matches the input directly, with no shift needed at all.
    EXPECT_LT(rmse_zero_phase, 0.05);
    EXPECT_LT(rmse_zero_phase, rmse_undelayed * 0.5);
}

// ---------------------------------------------------------------------------
// firwin() / firwin_highpass(): windowed-sinc FIR filter design.
// ---------------------------------------------------------------------------

TEST(FirWinTest, dc_gain_near_unity) {
    for (int n_taps : {15, 31, 63}) {
        for (double cutoff : {0.2, 0.4, 0.6}) {
            const auto taps = firwin(n_taps, cutoff, FirWindow::Hamming);
            ASSERT_EQ(taps.size(), static_cast<size_t>(n_taps));
            double sum = 0.0;
            for (double v : taps) {
                sum += v;
            }
            EXPECT_NEAR(sum, 1.0, 1e-9);
        }
    }
}

TEST(FirWinTest, symmetric_linear_phase) {
    for (int n_taps : {15, 31, 64}) {
        for (FirWindow w : {FirWindow::Rectangular, FirWindow::Hamming, FirWindow::Hann, FirWindow::Blackman}) {
            const auto taps = firwin(n_taps, 0.3, w);
            ASSERT_EQ(taps.size(), static_cast<size_t>(n_taps));
            for (size_t i = 0; i < taps.size(); ++i) {
                EXPECT_NEAR(taps[i], taps[taps.size() - 1 - i], 1e-9);
            }
        }
    }
}

TEST(FirWinTest, amplitude_response_lowpass_via_filtfilt) {
    // cutoff = 0.3 (normalized to Nyquist) => passband edge at 0.15 cycles/sample.
    const int n_taps = 101;
    const auto taps = firwin(n_taps, 0.3, FirWindow::Hamming);

    const size_t N = 400;
    const double f_low = 0.02;  // well below cutoff
    const double f_high = 0.4;  // well above cutoff
    std::vector<double> x_low(N), x_high(N);
    for (size_t i = 0; i < N; ++i) {
        x_low[i] = std::sin(2.0 * M_PI * f_low * static_cast<double>(i));
        x_high[i] = std::sin(2.0 * M_PI * f_high * static_cast<double>(i));
    }

    const auto y_low = filtfilt(taps, {1.0}, x_low);
    const auto y_high = filtfilt(taps, {1.0}, x_high);

    const size_t margin = 60;
    double amp_low = 0.0;
    double amp_high = 0.0;
    for (size_t i = margin; i + margin < N; ++i) {
        amp_low = std::max(amp_low, std::abs(y_low[i]));
        amp_high = std::max(amp_high, std::abs(y_high[i]));
    }

    EXPECT_NEAR(amp_low, 1.0, 0.1);   // near-unity passband gain
    EXPECT_LT(amp_high, 0.2);         // substantially attenuated stopband
}

TEST(FirWinTest, amplitude_response_second_cutoff) {
    const int n_taps = 81;
    const auto taps = firwin(n_taps, 0.5, FirWindow::Blackman);

    const size_t N = 400;
    const double f_low = 0.05;  // well below the 0.25 cycles/sample passband edge
    const double f_high = 0.45; // well above it
    std::vector<double> x_low(N), x_high(N);
    for (size_t i = 0; i < N; ++i) {
        x_low[i] = std::sin(2.0 * M_PI * f_low * static_cast<double>(i));
        x_high[i] = std::sin(2.0 * M_PI * f_high * static_cast<double>(i));
    }

    const auto y_low = filtfilt(taps, {1.0}, x_low);
    const auto y_high = filtfilt(taps, {1.0}, x_high);

    const size_t margin = 60;
    double amp_low = 0.0;
    double amp_high = 0.0;
    for (size_t i = margin; i + margin < N; ++i) {
        amp_low = std::max(amp_low, std::abs(y_low[i]));
        amp_high = std::max(amp_high, std::abs(y_high[i]));
    }

    EXPECT_NEAR(amp_low, 1.0, 0.1);
    EXPECT_LT(amp_high, 0.2);
}

TEST(FirWinTest, rectangular_window_has_worse_stopband_than_tapered) {
    const int n_taps = 51;
    const double cutoff = 0.3;
    const size_t N = 400;
    const double f_stop = 0.4; // well above cutoff, in the stopband

    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = std::sin(2.0 * M_PI * f_stop * static_cast<double>(i));
    }

    auto stopband_amplitude = [&](FirWindow w) {
        const auto taps = firwin(n_taps, cutoff, w);
        const auto y = filtfilt(taps, {1.0}, x);
        const size_t margin = 60;
        double amp = 0.0;
        for (size_t i = margin; i + margin < N; ++i) {
            amp = std::max(amp, std::abs(y[i]));
        }
        return amp;
    };

    const double amp_rect = stopband_amplitude(FirWindow::Rectangular);
    const double amp_hamming = stopband_amplitude(FirWindow::Hamming);
    const double amp_hann = stopband_amplitude(FirWindow::Hann);
    const double amp_blackman = stopband_amplitude(FirWindow::Blackman);

    EXPECT_GT(amp_rect, amp_hamming);
    EXPECT_GT(amp_rect, amp_hann);
    EXPECT_GT(amp_rect, amp_blackman);
}

TEST(FirWinTest, edge_cases) {
    EXPECT_TRUE(firwin(0, 0.3).empty());
    EXPECT_TRUE(firwin(-5, 0.3).empty());
    EXPECT_EQ(firwin(1, 0.3).size(), 1u);

    // Cutoff near the extremes should stay finite (no NaN/Inf) and not crash.
    const auto near_zero = firwin(21, 0.001, FirWindow::Hamming);
    const auto near_one = firwin(21, 0.999, FirWindow::Hamming);
    ASSERT_EQ(near_zero.size(), 21u);
    ASSERT_EQ(near_one.size(), 21u);
    for (double v : near_zero) {
        EXPECT_TRUE(std::isfinite(v));
    }
    for (double v : near_one) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(FirWinTest, highpass_complementary_identity) {
    for (int n_taps : {15, 31, 63}) {
        const auto lp = firwin(n_taps, 0.4, FirWindow::Hamming);
        const auto hp = firwin_highpass(n_taps, 0.4, FirWindow::Hamming);
        ASSERT_EQ(lp.size(), static_cast<size_t>(n_taps));
        ASSERT_EQ(hp.size(), static_cast<size_t>(n_taps));

        const size_t center = static_cast<size_t>(n_taps - 1) / 2;
        for (size_t i = 0; i < lp.size(); ++i) {
            if (i == center) {
                EXPECT_NEAR(lp[i] + hp[i], 1.0, 1e-9);
            } else {
                EXPECT_NEAR(lp[i] + hp[i], 0.0, 1e-9);
            }
        }
    }
}

TEST(FirWinTest, highpass_amplitude_response_is_inverted) {
    const int n_taps = 101;
    const auto taps = firwin_highpass(n_taps, 0.3, FirWindow::Hamming);
    ASSERT_EQ(taps.size(), static_cast<size_t>(n_taps));

    const size_t N = 400;
    const double f_low = 0.02;  // now in the highpass's stopband
    const double f_high = 0.4;  // now in the highpass's passband
    std::vector<double> x_low(N), x_high(N);
    for (size_t i = 0; i < N; ++i) {
        x_low[i] = std::sin(2.0 * M_PI * f_low * static_cast<double>(i));
        x_high[i] = std::sin(2.0 * M_PI * f_high * static_cast<double>(i));
    }

    const auto y_low = filtfilt(taps, {1.0}, x_low);
    const auto y_high = filtfilt(taps, {1.0}, x_high);

    const size_t margin = 60;
    double amp_low = 0.0;
    double amp_high = 0.0;
    for (size_t i = margin; i + margin < N; ++i) {
        amp_low = std::max(amp_low, std::abs(y_low[i]));
        amp_high = std::max(amp_high, std::abs(y_high[i]));
    }

    EXPECT_LT(amp_low, 0.2);          // low frequency now attenuated
    EXPECT_NEAR(amp_high, 1.0, 0.1);  // high frequency now passed
}

TEST(FirWinTest, highpass_edge_cases) {
    EXPECT_TRUE(firwin_highpass(0, 0.3).empty());
    EXPECT_TRUE(firwin_highpass(-3, 0.3).empty());
    EXPECT_TRUE(firwin_highpass(10, 0.3).empty()); // even n_taps rejected
    EXPECT_EQ(firwin_highpass(11, 0.3).size(), 11u);
}

// ---------------------------------------------------------------------------
// savgol(): Savitzky-Golay least-squares polynomial smoothing filter.
// ---------------------------------------------------------------------------

TEST(SavGolTest, known_kernel_window5_polyorder2_hand_computed) {
    // Published Savitzky-Golay coefficients for window_length=5, polyorder=2:
    // h = [-3, 12, 17, 12, -3] / 35 (e.g. Savitzky & Golay 1964 / standard references).
    const std::vector<double> expected{-3.0 / 35.0, 12.0 / 35.0, 17.0 / 35.0, 12.0 / 35.0, -3.0 / 35.0};

    // Apply the filter to a signal long enough that at least one interior point (index 2) has
    // a full centered window, and check it matches the hand-applied kernel exactly.
    const std::vector<double> x{2.0, -1.0, 3.0, 0.5, 4.0, -2.0, 1.5};
    const auto y = savgol(x, 5, 2);
    ASSERT_EQ(y.size(), x.size());

    double expected_center = 0.0;
    for (size_t j = 0; j < expected.size(); ++j) {
        expected_center += expected[j] * x[j];
    }
    EXPECT_NEAR(y[2], expected_center, 1e-12);

    // A second interior point, sliding the same kernel one sample forward.
    double expected_center2 = 0.0;
    for (size_t j = 0; j < expected.size(); ++j) {
        expected_center2 += expected[j] * x[j + 1];
    }
    EXPECT_NEAR(y[3], expected_center2, 1e-12);

    // Boundary points (no full centered window) are left unfiltered, i.e. copied from input.
    EXPECT_DOUBLE_EQ(y[0], x[0]);
    EXPECT_DOUBLE_EQ(y[1], x[1]);
    EXPECT_DOUBLE_EQ(y[5], x[5]);
    EXPECT_DOUBLE_EQ(y[6], x[6]);
}

TEST(SavGolTest, preserves_cubic_trend_better_than_raw_noisy_signal) {
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 5.0);

    // Large N keeps the empirical noise statistics close to their theoretical values so the
    // comparison below isn't sensitive to the particular fixed-seed noise realization.
    const size_t N = 400;
    std::vector<double> clean(N), noisy(N);
    for (size_t i = 0; i < N; ++i) {
        // f(t) = t^3 on a small, centered scale so the noise is meaningfully disruptive but the
        // cubic curvature dominates over the smoothing window.
        const double t = static_cast<double>(i) - static_cast<double>(N) / 2.0;
        clean[i] = t * t * t;
        noisy[i] = clean[i] + noise(rng);
    }

    const auto smoothed = savgol(noisy, 11, 3); // polyorder=3 matches the cubic exactly
    ASSERT_EQ(smoothed.size(), noisy.size());

    const size_t margin = 5; // (window_length-1)/2, skip unfiltered boundary points
    double err_smoothed = 0.0;
    double err_noisy = 0.0;
    for (size_t i = margin; i + margin < N; ++i) {
        err_smoothed += (smoothed[i] - clean[i]) * (smoothed[i] - clean[i]);
        err_noisy += (noisy[i] - clean[i]) * (noisy[i] - clean[i]);
    }
    const double rmse_smoothed = std::sqrt(err_smoothed / static_cast<double>(N - 2 * margin));
    const double rmse_noisy = std::sqrt(err_noisy / static_cast<double>(N - 2 * margin));

    // The polynomial order matches the underlying trend exactly, so the smoothed signal should
    // track it far more closely than the raw noisy samples (theoretically, residual variance is
    // reduced by a factor of sum(h_i^2) ~= 0.21 for window_length=11, polyorder=3, i.e. RMSE
    // should shrink by roughly sqrt(0.21) ~= 0.46; 0.7 leaves comfortable margin for sampling
    // noise while still requiring a clear, substantial improvement).
    EXPECT_LT(rmse_smoothed, rmse_noisy * 0.7);
}

TEST(SavGolTest, reduces_noise_variance_on_noisy_sine) {
    std::mt19937 rng(123);
    std::normal_distribution<double> noise(0.0, 0.3);

    const size_t N = 200;
    const double f = 0.02;
    std::vector<double> clean(N), noisy(N);
    for (size_t i = 0; i < N; ++i) {
        clean[i] = std::sin(2.0 * M_PI * f * static_cast<double>(i));
        noisy[i] = clean[i] + noise(rng);
    }

    const auto smoothed = savgol(noisy, 9, 2);
    ASSERT_EQ(smoothed.size(), noisy.size());

    const size_t margin = 4; // (window_length-1)/2
    double residual_var_smoothed = 0.0;
    double residual_var_noisy = 0.0;
    // Roughness proxy: sum of squared second differences, which is inflated by high-frequency
    // noise but small for the slowly-varying clean sine.
    double roughness_smoothed = 0.0;
    double roughness_noisy = 0.0;
    size_t count = 0;
    for (size_t i = margin; i + margin < N; ++i) {
        const double res_s = smoothed[i] - clean[i];
        const double res_n = noisy[i] - clean[i];
        residual_var_smoothed += res_s * res_s;
        residual_var_noisy += res_n * res_n;
        if (i + 1 + margin < N && i >= margin + 1) {
            const double d2_smoothed = smoothed[i + 1] - 2.0 * smoothed[i] + smoothed[i - 1];
            const double d2_noisy = noisy[i + 1] - 2.0 * noisy[i] + noisy[i - 1];
            roughness_smoothed += d2_smoothed * d2_smoothed;
            roughness_noisy += d2_noisy * d2_noisy;
        }
        ++count;
    }
    residual_var_smoothed /= static_cast<double>(count);
    residual_var_noisy /= static_cast<double>(count);

    EXPECT_LT(residual_var_smoothed, residual_var_noisy);
    EXPECT_LT(roughness_smoothed, roughness_noisy);
}

TEST(SavGolTest, edge_cases_return_empty) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    EXPECT_TRUE(savgol(x, 4, 2).empty());   // even window_length
    EXPECT_TRUE(savgol(x, 0, 0).empty());   // non-positive window_length
    EXPECT_TRUE(savgol(x, -3, 1).empty());  // negative window_length
    EXPECT_TRUE(savgol(x, 5, 5).empty());   // polyorder >= window_length
    EXPECT_TRUE(savgol(x, 5, 6).empty());   // polyorder > window_length
    EXPECT_TRUE(savgol(x, 5, -1).empty());  // negative polyorder
    EXPECT_TRUE(savgol(x, 11, 2).empty());  // window_length > x.size()
    EXPECT_TRUE(savgol(std::vector<double>{}, 5, 2).empty()); // empty input

    // Sanity: a valid, small configuration on this same signal should NOT be empty.
    EXPECT_FALSE(savgol(x, 5, 2).empty());
}

// ---------------------------------------------------------------------------
// median_filter(): nonlinear sliding-window median denoising filter.
// ---------------------------------------------------------------------------

TEST(MedianFilterTest, known_small_example_hand_computed) {
    // x = {5, 1, 3, 2, 4}, window_length=3.
    // center=1: window {5,1,3} sorted -> {1,3,5}, median 3.
    // center=2: window {1,3,2} sorted -> {1,2,3}, median 2.
    // center=3: window {3,2,4} sorted -> {2,3,4}, median 3.
    // Boundary points (half=1): y[0]=x[0]=5, y[4]=x[4]=4.
    const std::vector<double> x{5.0, 1.0, 3.0, 2.0, 4.0};
    const auto y = median_filter(x, 3);
    ASSERT_EQ(y.size(), x.size());
    const std::vector<double> expected{5.0, 3.0, 2.0, 3.0, 4.0};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(y[i], expected[i]);
    }
}

TEST(MedianFilterTest, window_length_one_is_identity) {
    const std::vector<double> x{5.0, -1.0, 3.0, 2.0, 4.0, 0.0, -7.5};
    const auto y = median_filter(x, 1);
    ASSERT_EQ(y.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_DOUBLE_EQ(y[i], x[i]);
    }
}

TEST(MedianFilterTest, removes_single_outlier_spike_completely) {
    // Constant signal with one huge outlier spike.
    std::vector<double> x(11, 2.0);
    x[5] = 1.0e6; // sharp impulsive outlier, far outside the local range

    const auto y = median_filter(x, 5);
    ASSERT_EQ(y.size(), x.size());

    // The median of a window centered on the spike (four 2.0's and the spike) is 2.0: the
    // spike is completely rejected, not merely attenuated.
    EXPECT_DOUBLE_EQ(y[5], 2.0);
    // Neighboring interior points, whose windows also contain only 2.0 or the single spike,
    // are unaffected too.
    for (size_t i = 2; i + 2 < x.size(); ++i) {
        EXPECT_DOUBLE_EQ(y[i], 2.0);
    }
}

TEST(MedianFilterTest, robust_to_outlier_unlike_moving_average) {
    // Same spiked signal, contrast median_filter against moving_average directly: a moving
    // average is pulled arbitrarily far toward the outlier, while the median completely
    // rejects it.
    std::vector<double> x(11, 2.0);
    x[5] = 1.0e6;

    const auto med = median_filter(x, 5);
    const auto avg = moving_average(x, 5);
    ASSERT_EQ(med.size(), x.size());
    ASSERT_EQ(avg.size(), x.size());

    EXPECT_DOUBLE_EQ(med[5], 2.0);
    // moving_average's window ending at i=5 (indices 1..5) includes the spike, so it is
    // dragged far away from the surrounding baseline of 2.0.
    EXPECT_GT(avg[5], 1000.0);
    EXPECT_GT(std::abs(avg[5] - 2.0), std::abs(med[5] - 2.0) * 1000.0);
}

TEST(MedianFilterTest, boundary_points_copied_unfiltered) {
    const std::vector<double> x{9.0, 8.0, 1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0};
    const int window_length = 5;
    const auto y = median_filter(x, window_length);
    ASSERT_EQ(y.size(), x.size());

    const size_t half = static_cast<size_t>((window_length - 1) / 2);
    for (size_t i = 0; i < half; ++i) {
        EXPECT_DOUBLE_EQ(y[i], x[i]);
    }
    for (size_t i = x.size() - half; i < x.size(); ++i) {
        EXPECT_DOUBLE_EQ(y[i], x[i]);
    }
}

TEST(MedianFilterTest, edge_cases_return_empty) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    EXPECT_TRUE(median_filter(x, 4).empty());   // even window_length
    EXPECT_TRUE(median_filter(x, 0).empty());   // non-positive window_length
    EXPECT_TRUE(median_filter(x, -3).empty());  // negative window_length
    EXPECT_TRUE(median_filter(x, 11).empty());  // window_length > x.size()
    EXPECT_TRUE(median_filter(std::vector<double>{}, 3).empty()); // empty input

    // Sanity: a valid, small configuration on this same signal should NOT be empty.
    EXPECT_FALSE(median_filter(x, 3).empty());
}

TEST(MedianFilterTest, constant_signal_filters_to_itself) {
    const std::vector<double> x(15, 3.5);
    for (int window_length : {1, 3, 5, 9, 15}) {
        const auto y = median_filter(x, window_length);
        ASSERT_EQ(y.size(), x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            EXPECT_DOUBLE_EQ(y[i], 3.5);
        }
    }
}

TEST(MedianFilterTest, monotonic_ramp_interior_points_unchanged) {
    // For a monotonically increasing signal of consecutive integers, the median of any
    // symmetric window centered at i is exactly x[i] (the center value ranks in the middle).
    std::vector<double> x(20);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<double>(i + 1); // 1, 2, ..., 20
    }

    const int window_length = 7;
    const auto y = median_filter(x, window_length);
    ASSERT_EQ(y.size(), x.size());

    const size_t half = static_cast<size_t>((window_length - 1) / 2);
    for (size_t i = half; i + half < x.size(); ++i) {
        EXPECT_DOUBLE_EQ(y[i], x[i]);
    }
}

TEST(MedianFilterTest, output_length_matches_input_for_various_windows) {
    const std::vector<double> x{4.0, 2.0, 9.0, 1.0, 7.0, 3.0, 8.0, 5.0, 6.0, 0.0, 10.0};
    for (int window_length : {1, 3, 5, 7, 9, 11}) {
        const auto y = median_filter(x, window_length);
        EXPECT_EQ(y.size(), x.size());
    }
}

TEST(MedianFilterTest, window_length_three_never_moves_output_outside_window_range) {
    // General sanity property: every filtered interior sample must lie within the min/max of
    // its own window (a median is always within [min, max] of the values it's drawn from).
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    std::vector<double> x(50);
    for (double& v : x) {
        v = dist(rng);
    }

    const int window_length = 5;
    const auto y = median_filter(x, window_length);
    ASSERT_EQ(y.size(), x.size());

    const size_t half = static_cast<size_t>((window_length - 1) / 2);
    for (size_t center = half; center + half < x.size(); ++center) {
        double window_min = x[center - half];
        double window_max = x[center - half];
        for (size_t j = center - half; j <= center + half; ++j) {
            window_min = std::min(window_min, x[j]);
            window_max = std::max(window_max, x[j]);
        }
        EXPECT_GE(y[center], window_min);
        EXPECT_LE(y[center], window_max);
    }
}

TEST(MedianFilterTest, two_outlier_spikes_within_one_window_still_rejected) {
    // Window_length=5 window has 5 slots; with only 2 outliers in any given window, the
    // majority (3) baseline values still determine the median.
    std::vector<double> x(13, 1.0);
    x[5] = -1000.0;
    x[6] = 1000.0;

    const auto y = median_filter(x, 5);
    ASSERT_EQ(y.size(), x.size());
    // Any window of length 5 containing both spikes still has at least 3 baseline (1.0)
    // values, so the median remains exactly 1.0.
    for (size_t center = 5; center <= 6; ++center) {
        EXPECT_DOUBLE_EQ(y[center], 1.0);
    }
}
