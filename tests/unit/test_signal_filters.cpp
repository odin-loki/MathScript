#include <gtest/gtest.h>

#include "ms/signal/signal.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <random>
#include <span>

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

// ---------------------------------------------------------------------------
// filter() DF2T: impulse/step response, scipy reference, sosfilt consistency.
// ---------------------------------------------------------------------------

TEST(SignalFilterDf2tTest, fir_impulse_response_equals_b_taps) {
    // scipy.signal.lfilter([0.25, 0.5, 0.25], [1], unit impulse)
    const std::vector<double> b{0.25, 0.5, 0.25};
    const std::vector<double> a{1.0};
    std::vector<double> impulse(16, 0.0);
    impulse[0] = 1.0;
    const auto y = filter(b, a, impulse);
    const std::vector<double> expected{0.25, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0};
    ASSERT_GE(y.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y[i], expected[i], 1e-12);
    }
}

TEST(SignalFilterDf2tTest, iir_impulse_response_matches_scipy_reference) {
    // scipy.signal.lfilter on 2nd-order Butterworth lowpass (Wn=0.25, fs=2).
    const std::vector<double> b{0.09763107, 0.19526215, 0.09763107};
    const std::vector<double> a{1.0, -0.94280904, 0.33333333};
    std::vector<double> impulse(16, 0.0);
    impulse[0] = 1.0;
    const auto y = filter(b, a, impulse);
    const std::vector<double> expected{
        0.09763107, 0.28730961, 0.33596547, 0.22098142,
        0.09635479, 0.01718369, -0.01591732, -0.02073489,
    };
    ASSERT_GE(y.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y[i], expected[i], 1e-6);
    }
}

TEST(SignalFilterDf2tTest, fir_step_response_hand_computed) {
    const std::vector<double> b{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
    const std::vector<double> a{1.0};
    const std::vector<double> step(12, 1.0);
    const auto y = filter(b, a, step);
    const std::vector<double> expected{1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    ASSERT_EQ(y.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y[i], expected[i], 1e-12);
    }
}

TEST(SignalFilterDf2tTest, iir_step_response_matches_scipy_reference) {
    // scipy.signal.lfilter([0.5], [1, -0.5], step starting at index 3)
    const std::vector<double> b{0.5};
    const std::vector<double> a{1.0, -0.5};
    const std::vector<double> step{0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const auto y = filter(b, a, step);
    const std::vector<double> expected_tail{0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.9921875};
    ASSERT_EQ(y.size(), step.size());
    for (size_t i = 0; i < expected_tail.size(); ++i) {
        EXPECT_NEAR(y[i + 3], expected_tail[i], 1e-10);
    }
}

TEST(SignalFilterDf2tTest, butterworth_signal_matches_scipy_lfilter) {
    // scipy.signal.lfilter(b, a, [1,0,-1,0,1,0,-1,0]) — same coeffs as sosfilt reference test.
    const std::vector<double> b{0.09763107, 0.19526215, 0.09763107};
    const std::vector<double> a{1.0, -0.94280904, 0.33333333};
    const std::vector<double> x{1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0};
    const auto y = filter(b, a, x);
    const std::vector<double> expected{
        0.09763107, 0.2873096, 0.2383344, -0.06632819,
        -0.14197961, 0.08351188, 0.12606229, -0.10424677,
    };
    ASSERT_EQ(y.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y[i], expected[i], 1e-6);
    }
}

TEST(SignalFilterDf2tTest, fourth_order_direct_matches_sosfilt_cascade) {
    // Direct-form coeffs from convolving two 2nd-order Butterworth SOS sections.
    const std::vector<double> b{0.01020948, 0.04083792, 0.06125688, 0.04083792, 0.01020948};
    const std::vector<double> a{1.0, -1.96842778, 1.73586071, -0.72447083, 0.1203896};
    const std::vector<std::array<double, 6>> sos{
        {0.01020948, 0.02041896, 0.01020948, 1.0, -0.85539793, 0.20971536},
        {1.0, 2.0, 1.0, 1.0, -1.11302985, 0.57406192},
    };
    const std::vector<double> x{0.0, 1.0, 0.5, -0.25, 0.75, -1.0, 0.3, 0.6, -0.4, 0.2,
                                0.9, -0.1, 0.4, -0.6, 0.8, 0.0};
    const auto y_direct = filter(b, a, x);
    const auto y_sos = sosfilt(sos, x);
    ASSERT_EQ(y_direct.size(), x.size());
    ASSERT_EQ(y_sos.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(y_direct[i], y_sos[i], 1e-8);
    }
}

TEST(SignalFilterDf2tTest, fourth_order_direct_matches_scipy_lfilter) {
    // scipy.signal.lfilter on 4th-order Butterworth (product of two SOS sections).
    const std::vector<double> b{0.01020948, 0.04083792, 0.06125688, 0.04083792, 0.01020948};
    const std::vector<double> a{1.0, -1.96842778, 1.73586071, -0.72447083, 0.1203896};
    const std::vector<double> x{0.0, 1.0, 0.5, -0.25, 0.75, -1.0, 0.3, 0.6};
    const auto y = filter(b, a, x);
    const std::vector<double> expected{
        0.0, 0.01020948, 0.06603928, 0.1913948, 0.3384223, 0.41627274, 0.36572121, 0.20463569,
    };
    ASSERT_EQ(y.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y[i], expected[i], 1e-6);
    }
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

namespace {

std::vector<double> median_filter_reference(const std::vector<double>& x, int window_length) {
    if (window_length <= 0 || window_length % 2 == 0) {
        return {};
    }
    if (x.size() < static_cast<size_t>(window_length)) {
        return {};
    }

    const size_t half = static_cast<size_t>((window_length - 1) / 2);
    const size_t n = x.size();

    std::vector<double> y = x;
    std::vector<double> window(static_cast<size_t>(window_length));
    for (size_t center = half; center + half < n; ++center) {
        for (size_t j = 0; j < window.size(); ++j) {
            window[j] = x[center - half + j];
        }
        std::sort(window.begin(), window.end());
        y[center] = window[half];
    }
    return y;
}

} // namespace

TEST(MedianFilterTest, window_three_hand_computed_even_spread) {
    // x = {10, 2, 8, 4, 6}, window=3.
    // center=1: {10,2,8} -> median 8; center=2: {2,8,4} -> 4; center=3: {8,4,6} -> 6
    const std::vector<double> x{10.0, 2.0, 8.0, 4.0, 6.0};
    const auto y = median_filter(x, 3);
    ASSERT_EQ(y.size(), x.size());
    const std::vector<double> expected{10.0, 8.0, 4.0, 6.0, 6.0};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(y[i], expected[i]);
    }
}

TEST(MedianFilterTest, window_five_hand_computed) {
    // x = {9, 1, 7, 3, 5, 2, 8}, window=5.
    // center=2: {9,1,7,3,5} sorted {1,3,5,7,9} median 5
    // center=3: {1,7,3,5,2} sorted {1,2,3,5,7} median 3
    // center=4: {7,3,5,2,8} sorted {2,3,5,7,8} median 5
    const std::vector<double> x{9.0, 1.0, 7.0, 3.0, 5.0, 2.0, 8.0};
    const auto y = median_filter(x, 5);
    ASSERT_EQ(y.size(), x.size());
    const std::vector<double> expected{9.0, 1.0, 5.0, 3.0, 5.0, 2.0, 8.0};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(y[i], expected[i]);
    }
}

TEST(MedianFilterTest, window_seven_hand_computed) {
    const std::vector<double> x{6.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0, 4.0, 5.0};
    const auto y = median_filter(x, 7);
    ASSERT_EQ(y.size(), x.size());

    const std::vector<double> expected{6.0, 2.0, 8.0, 6.0, 4.0, 5.0, 7.0, 4.0, 5.0};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(y[i], expected[i]);
    }
}

TEST(MedianFilterTest, large_n_matches_reference_implementation) {
    std::mt19937 rng(219);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    std::vector<double> x(10000);
    for (double& v : x) {
        v = dist(rng);
    }

    for (int window_length : {3, 5, 7, 15, 31, 101}) {
        const auto fast = median_filter(x, window_length);
        const auto ref = median_filter_reference(x, window_length);
        ASSERT_EQ(fast.size(), ref.size());
        for (size_t i = 0; i < ref.size(); ++i) {
            EXPECT_DOUBLE_EQ(fast[i], ref[i]) << "window_length=" << window_length << " i=" << i;
        }
    }
}

TEST(MedianFilterTest, sliding_multiset_path_handles_duplicate_values) {
    const std::vector<double> x{5.0, 5.0, 1.0, 5.0, 5.0, 5.0, 1.0, 5.0, 5.0};
    const auto y = median_filter(x, 7);
    const auto ref = median_filter_reference(x, 7);
    ASSERT_EQ(y.size(), ref.size());
    for (size_t i = 0; i < ref.size(); ++i) {
        EXPECT_DOUBLE_EQ(y[i], ref[i]);
    }
}

TEST(MedianFilterTest, window_seven_random_matches_reference) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 20);
    std::vector<double> x(500);
    for (double& v : x) {
        v = static_cast<double>(dist(rng));
    }

    const auto y = median_filter(x, 7);
    const auto ref = median_filter_reference(x, 7);
    for (size_t i = 0; i < ref.size(); ++i) {
        EXPECT_DOUBLE_EQ(y[i], ref[i]) << "i=" << i;
    }
}

TEST(MedianFilterTest, large_window_odd_size_preserves_constant_signal) {
    std::vector<double> x(50000, -2.25);
    const auto y = median_filter(x, 401);
    ASSERT_EQ(y.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_DOUBLE_EQ(y[i], -2.25);
    }
}

// ---------------------------------------------------------------------------
// lms_adaptive_filter(): online LMS adaptive FIR filter.
// ---------------------------------------------------------------------------

namespace {

// Deterministic "random-ish" input signal: sum of a few incommensurate sinusoids plus a
// simple congruential noise term, used as a stand-in for a broadband excitation signal (LMS
// needs a spectrally-rich input to identify all taps of the unknown system).
std::vector<double> lms_test_signal(size_t n) {
    std::vector<double> x(n);
    uint32_t state = 12345u;
    for (size_t i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        const double noise = (static_cast<double>(state % 10000u) / 10000.0 - 0.5) * 0.2;
        x[i] = std::sin(2.0 * M_PI * 0.017 * static_cast<double>(i)) +
               0.6 * std::sin(2.0 * M_PI * 0.083 * static_cast<double>(i)) + noise;
    }
    return x;
}

// Applies a fixed causal FIR filter (true_weights) directly, i.e.
// d[n] = sum_k true_weights[k] * x[n-k], with zero-padding for n-k < 0. Used to build the
// "desired" reference signal for an LMS system-identification scenario.
std::vector<double> apply_known_fir(const std::vector<double>& x,
                                     const std::vector<double>& true_weights) {
    std::vector<double> d(x.size(), 0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        double acc = 0.0;
        for (size_t k = 0; k < true_weights.size(); ++k) {
            if (i >= k) {
                acc += true_weights[k] * x[i - k];
            }
        }
        d[i] = acc;
    }
    return d;
}

} // namespace

TEST(LMSAdaptiveFilterTest, converges_to_known_true_weights) {
    const std::vector<double> true_weights{0.5, 0.3, -0.2};
    const size_t N = 20000;
    const auto x = lms_test_signal(N);
    const auto d = apply_known_fir(x, true_weights);

    const auto result = lms_adaptive_filter(x, d, 3, 0.01);
    ASSERT_EQ(result.weights.size(), true_weights.size());
    ASSERT_EQ(result.output.size(), N);
    ASSERT_EQ(result.error.size(), N);

    for (size_t k = 0; k < true_weights.size(); ++k) {
        EXPECT_NEAR(result.weights[k], true_weights[k], 0.05);
    }
}

TEST(LMSAdaptiveFilterTest, error_decreases_substantially_over_time) {
    const std::vector<double> true_weights{0.5, 0.3, -0.2};
    const size_t N = 4000;
    const auto x = lms_test_signal(N);
    const auto d = apply_known_fir(x, true_weights);

    const auto result = lms_adaptive_filter(x, d, 3, 0.01);
    ASSERT_EQ(result.error.size(), N);

    const size_t window = 200;
    double mse_early = 0.0;
    for (size_t i = 0; i < window; ++i) {
        mse_early += result.error[i] * result.error[i];
    }
    mse_early /= static_cast<double>(window);

    double mse_late = 0.0;
    for (size_t i = N - window; i < N; ++i) {
        mse_late += result.error[i] * result.error[i];
    }
    mse_late /= static_cast<double>(window);

    EXPECT_LT(mse_late, mse_early * 0.1);
}

TEST(LMSAdaptiveFilterTest, zero_input_and_desired_converges_trivially) {
    const std::vector<double> x(50, 0.0);
    const std::vector<double> d(50, 0.0);

    const auto result = lms_adaptive_filter(x, d, 4, 0.1);
    ASSERT_EQ(result.weights.size(), 4u);
    for (double w : result.weights) {
        EXPECT_DOUBLE_EQ(w, 0.0);
    }
    for (double y : result.output) {
        EXPECT_DOUBLE_EQ(y, 0.0);
    }
    for (double e : result.error) {
        EXPECT_DOUBLE_EQ(e, 0.0);
    }
}

TEST(LMSAdaptiveFilterTest, mu_zero_means_no_adaptation) {
    const size_t N = 30;
    const auto x = lms_test_signal(N);
    const std::vector<double> true_weights{0.4, -0.1};
    const auto d = apply_known_fir(x, true_weights);

    const auto result = lms_adaptive_filter(x, d, 2, 0.0);
    ASSERT_EQ(result.weights.size(), 2u);

    // Weights never move away from their zero initial value.
    EXPECT_DOUBLE_EQ(result.weights[0], 0.0);
    EXPECT_DOUBLE_EQ(result.weights[1], 0.0);

    // Output is computed from all-zero weights, so it's all-zero too.
    ASSERT_EQ(result.output.size(), N);
    for (double y : result.output) {
        EXPECT_DOUBLE_EQ(y, 0.0);
    }

    // Error exactly equals the desired signal at every step (e[n] = d[n] - 0).
    ASSERT_EQ(result.error.size(), N);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(result.error[i], d[i]);
    }
}

TEST(LMSAdaptiveFilterTest, mismatched_lengths_returns_empty_result) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> d{1.0, 2.0, 3.0};

    const auto result = lms_adaptive_filter(x, d, 2, 0.01);
    EXPECT_TRUE(result.output.empty());
    EXPECT_TRUE(result.error.empty());
    EXPECT_TRUE(result.weights.empty());
}

TEST(LMSAdaptiveFilterTest, non_positive_filter_length_returns_empty_result) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> d{1.0, 2.0, 3.0, 4.0};

    const auto zero_len = lms_adaptive_filter(x, d, 0, 0.01);
    EXPECT_TRUE(zero_len.output.empty());
    EXPECT_TRUE(zero_len.error.empty());
    EXPECT_TRUE(zero_len.weights.empty());

    const auto neg_len = lms_adaptive_filter(x, d, -3, 0.01);
    EXPECT_TRUE(neg_len.output.empty());
    EXPECT_TRUE(neg_len.error.empty());
    EXPECT_TRUE(neg_len.weights.empty());
}

TEST(LMSAdaptiveFilterTest, signal_shorter_than_filter_length_returns_empty_result) {
    const std::vector<double> x{1.0, 2.0};
    const std::vector<double> d{1.0, 2.0};

    const auto result = lms_adaptive_filter(x, d, 5, 0.01);
    EXPECT_TRUE(result.output.empty());
    EXPECT_TRUE(result.error.empty());
    EXPECT_TRUE(result.weights.empty());
}

TEST(LMSAdaptiveFilterTest, empty_input_returns_empty_result) {
    const auto result = lms_adaptive_filter({}, {}, 3, 0.01);
    EXPECT_TRUE(result.output.empty());
    EXPECT_TRUE(result.error.empty());
    EXPECT_TRUE(result.weights.empty());
}

TEST(LMSAdaptiveFilterTest, single_tap_learns_scalar_gain) {
    // x all equal to 2.0, d all equal to 6.0: the ideal single-tap gain is d/x = 3.0.
    const size_t N = 3000;
    const std::vector<double> x(N, 2.0);
    const std::vector<double> d(N, 6.0);

    const auto result = lms_adaptive_filter(x, d, 1, 0.01);
    ASSERT_EQ(result.weights.size(), 1u);
    EXPECT_NEAR(result.weights[0], 3.0, 0.05);

    // Output should converge close to the desired constant too.
    ASSERT_EQ(result.output.size(), N);
    EXPECT_NEAR(result.output.back(), 6.0, 0.5);
}

TEST(LMSAdaptiveFilterTest, single_tap_output_and_error_consistency) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> d{2.0, 4.0, 6.0, 8.0, 10.0};

    const auto result = lms_adaptive_filter(x, d, 1, 0.05);
    ASSERT_EQ(result.output.size(), x.size());
    ASSERT_EQ(result.error.size(), x.size());
    ASSERT_EQ(result.weights.size(), 1u);

    // First sample: weight starts at 0, so y[0] = 0*x[0] = 0, e[0] = d[0] - 0 = d[0].
    EXPECT_DOUBLE_EQ(result.output[0], 0.0);
    EXPECT_DOUBLE_EQ(result.error[0], d[0]);

    // Every sample must satisfy the defining relation e[n] = d[n] - y[n].
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(result.error[i], d[i] - result.output[i], 1e-12);
    }
}

TEST(LMSAdaptiveFilterTest, first_sample_output_is_always_zero_from_zero_initial_weights) {
    // At n=0, only w[0] (still 0) can contribute (zero-padding for k>0), so y[0] must be 0
    // regardless of filter_length or the input's value at n=0.
    const std::vector<double> x{7.0, -3.0, 2.5, 0.0, 1.0, -1.0, 4.0, 8.0};
    const std::vector<double> d{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    const auto result = lms_adaptive_filter(x, d, 4, 0.02);
    ASSERT_FALSE(result.output.empty());
    EXPECT_DOUBLE_EQ(result.output[0], 0.0);
}

TEST(LMSAdaptiveFilterTest, weight_vector_has_exactly_filter_length_entries) {
    const std::vector<double> x(20, 1.0);
    const std::vector<double> d(20, 1.0);
    for (int filter_length : {1, 2, 5, 10}) {
        const auto result = lms_adaptive_filter(x, d, filter_length, 0.01);
        EXPECT_EQ(result.weights.size(), static_cast<size_t>(filter_length));
    }
}

TEST(LMSAdaptiveFilterTest, negative_mu_still_updates_weights_deterministically) {
    // mu need not be positive for the mechanics of the update rule to be well-defined; a
    // negative mu simply moves weights in the opposite direction of the gradient step. This
    // test only checks the implementation applies the documented update rule literally,
    // regardless of the sign of mu (stability/convergence for bad mu values is the caller's
    // responsibility per the documented contract).
    const std::vector<double> x{1.0, 1.0, 1.0};
    const std::vector<double> d{2.0, 2.0, 2.0};
    const double mu = -0.1;

    const auto result = lms_adaptive_filter(x, d, 1, mu);
    ASSERT_EQ(result.weights.size(), 1u);

    // Hand-computed trace: w0 = 0.
    // n=0: y=0, e=2, w += mu*e*x[0] = -0.1*2*1 = -0.2 -> w=-0.2
    // n=1: y=w*x[1]=-0.2, e=2.2, w += mu*e*x[1] = -0.1*2.2*1=-0.22 -> w=-0.42
    // n=2: y=w*x[2]=-0.42, e=2.42, w += mu*e*x[2] = -0.1*2.42*1=-0.242 -> w=-0.662
    EXPECT_NEAR(result.weights[0], -0.662, 1e-9);
}

TEST(LMSAdaptiveFilterTest, two_tap_hand_computed_first_two_steps) {
    // filter_length=2, mu=0.1, x={1,2,3}, d={1,1,1}. Weights start at {0,0}.
    // n=0: y = w0*x[0] + w1*x[-1](=0) = 0. e = 1 - 0 = 1.
    //      w0 += mu*e*x[0] = 0.1*1*1 = 0.1 -> w0=0.1
    //      w1 += mu*e*x[-1] = 0.1*1*0 = 0   -> w1=0.0
    // n=1: y = w0*x[1] + w1*x[0] = 0.1*2 + 0.0*1 = 0.2. e = 1 - 0.2 = 0.8.
    //      w0 += mu*e*x[1] = 0.1*0.8*2 = 0.16 -> w0 = 0.26
    //      w1 += mu*e*x[0] = 0.1*0.8*1 = 0.08 -> w1 = 0.08
    const std::vector<double> x{1.0, 2.0, 3.0};
    const std::vector<double> d{1.0, 1.0, 1.0};
    const auto result = lms_adaptive_filter(x, d, 2, 0.1);

    ASSERT_EQ(result.output.size(), 3u);
    ASSERT_EQ(result.error.size(), 3u);
    EXPECT_NEAR(result.output[0], 0.0, 1e-12);
    EXPECT_NEAR(result.error[0], 1.0, 1e-12);
    EXPECT_NEAR(result.output[1], 0.2, 1e-12);
    EXPECT_NEAR(result.error[1], 0.8, 1e-12);
}

// ---------------------------------------------------------------------------
// cheby1(): Chebyshev Type I IIR filter design.
// ---------------------------------------------------------------------------

namespace {

double freqz_mag_at_z(const std::vector<double>& b, const std::vector<double>& a, double z_real) {
    std::complex<double> num{0.0, 0.0};
    std::complex<double> den{0.0, 0.0};
    const std::complex<double> z{z_real, 0.0};
    for (size_t i = 0; i < b.size(); ++i) {
        num += b[i] * std::pow(z, -static_cast<int>(i));
    }
    for (size_t i = 0; i < a.size(); ++i) {
        den += a[i] * std::pow(z, -static_cast<int>(i));
    }
    return std::abs(num / den);
}

double freqz_mag_dc(const std::vector<double>& b, const std::vector<double>& a) {
    return freqz_mag_at_z(b, a, 1.0);
}

double freqz_mag_nyquist(const std::vector<double>& b, const std::vector<double>& a) {
    return freqz_mag_at_z(b, a, -1.0);
}

} // namespace

TEST(SignalCheby1Test, invalid_order_returns_empty) {
    const auto coeffs = cheby1(0, 1.0, 100.0, 1000.0);
    EXPECT_TRUE(coeffs.b.empty());
    EXPECT_TRUE(coeffs.a.empty());
}

TEST(SignalCheby1Test, invalid_cutoff_returns_empty) {
    EXPECT_TRUE(cheby1(2, 1.0, 0.0, 1000.0).b.empty());
    EXPECT_TRUE(cheby1(2, 1.0, -10.0, 1000.0).b.empty());
    EXPECT_TRUE(cheby1(2, 1.0, 600.0, 1000.0).b.empty());
}

TEST(SignalCheby1Test, invalid_fs_or_ripple_returns_empty) {
    EXPECT_TRUE(cheby1(2, 1.0, 100.0, 0.0).b.empty());
    EXPECT_TRUE(cheby1(2, -1.0, 100.0, 1000.0).b.empty());
}

TEST(SignalCheby1Test, lowpass_second_order_matches_scipy_reference) {
    // scipy.signal.cheby1(2, 1.0, 0.25, fs=2.0, btype='low')
    const auto coeffs = cheby1(2, 1.0, 0.25, 2.0, FilterType::Lowpass);
    ASSERT_EQ(coeffs.b.size(), 3u);
    ASSERT_EQ(coeffs.a.size(), 3u);
    EXPECT_NEAR(coeffs.b[0], 0.10255744, 1e-6);
    EXPECT_NEAR(coeffs.b[1], 0.20511488, 1e-6);
    EXPECT_NEAR(coeffs.b[2], 0.10255744, 1e-6);
    EXPECT_NEAR(coeffs.a[0], 1.0, 1e-12);
    EXPECT_NEAR(coeffs.a[1], -0.98650792, 1e-6);
    EXPECT_NEAR(coeffs.a[2], 0.44679329, 1e-6);
}

TEST(SignalCheby1Test, highpass_second_order_matches_scipy_reference) {
    // scipy.signal.cheby1(2, 1.0, 0.25, fs=2.0, btype='high')
    const auto coeffs = cheby1(2, 1.0, 0.25, 2.0, FilterType::Highpass);
    ASSERT_EQ(coeffs.b.size(), 3u);
    ASSERT_EQ(coeffs.a.size(), 3u);
    EXPECT_NEAR(coeffs.b[0], 0.56838555, 1e-6);
    EXPECT_NEAR(coeffs.b[1], -1.13677109, 1e-6);
    EXPECT_NEAR(coeffs.b[2], 0.56838555, 1e-6);
    EXPECT_NEAR(coeffs.a[0], 1.0, 1e-12);
    EXPECT_NEAR(coeffs.a[1], -1.07698798, 1e-6);
    EXPECT_NEAR(coeffs.a[2], 0.4739683, 1e-6);
}

TEST(SignalCheby1Test, lowpass_dc_magnitude_exceeds_nyquist) {
    const auto coeffs = cheby1(4, 3.0, 100.0, 1000.0, FilterType::Lowpass);
    ASSERT_FALSE(coeffs.b.empty());
    ASSERT_FALSE(coeffs.a.empty());
    const double dc = freqz_mag_dc(coeffs.b, coeffs.a);
    const double nyq = freqz_mag_nyquist(coeffs.b, coeffs.a);
    EXPECT_GT(dc, nyq);
    EXPECT_GT(dc, 0.1);
    EXPECT_LT(nyq, 1e-6);
}

TEST(SignalCheby1Test, highpass_nyquist_exceeds_dc) {
    const auto coeffs = cheby1(4, 3.0, 100.0, 1000.0, FilterType::Highpass);
    ASSERT_FALSE(coeffs.b.empty());
    const double dc = freqz_mag_dc(coeffs.b, coeffs.a);
    const double nyq = freqz_mag_nyquist(coeffs.b, coeffs.a);
    EXPECT_LT(dc, 1e-5);
    EXPECT_GT(nyq, 0.5);
}

TEST(SignalCheby1Test, denominator_leading_coefficient_is_one) {
    for (const FilterType type : {FilterType::Lowpass, FilterType::Highpass}) {
        const auto coeffs = cheby1(3, 2.0, 250.0, 1000.0, type);
        ASSERT_FALSE(coeffs.a.empty());
        EXPECT_NEAR(coeffs.a[0], 1.0, 1e-12);
    }
}

TEST(SignalCheby1Test, filter_applies_designed_coefficients) {
    const auto coeffs = cheby1(2, 1.0, 0.25, 2.0);
    const std::vector<double> x{1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0};
    const auto y = filter(coeffs.b, coeffs.a, x);
    ASSERT_EQ(y.size(), x.size());
    for (double v : y) {
        EXPECT_TRUE(std::isfinite(v));
    }
    EXPECT_GT(std::abs(y.back()), 0.0);
}

TEST(SignalCheby1Test, fourth_order_lowpass_matches_scipy_reference) {
    // scipy.signal.cheby1(4, 3.0, 100.0, fs=1000.0, btype='low')
    const auto coeffs = cheby1(4, 3.0, 100.0, 1000.0, FilterType::Lowpass);
    ASSERT_EQ(coeffs.b.size(), 5u);
    ASSERT_EQ(coeffs.a.size(), 5u);
    EXPECT_NEAR(coeffs.b[0], 0.00105139, 1e-5);
    EXPECT_NEAR(coeffs.b[4], 0.00105139, 1e-5);
    EXPECT_NEAR(coeffs.a[1], -3.26916646, 1e-5);
    EXPECT_NEAR(coeffs.a[4], 0.69455867, 1e-5);
}

// ---------------------------------------------------------------------------
// cheby2(): Chebyshev Type II IIR filter design.
// ---------------------------------------------------------------------------

TEST(SignalCheby2Test, invalid_order_returns_empty) {
    const auto coeffs = cheby2(0, 40.0, 100.0, 1000.0);
    EXPECT_TRUE(coeffs.b.empty());
    EXPECT_TRUE(coeffs.a.empty());
}

TEST(SignalCheby2Test, invalid_cutoff_returns_empty) {
    EXPECT_TRUE(cheby2(2, 40.0, 0.0, 1000.0).b.empty());
    EXPECT_TRUE(cheby2(2, 40.0, -10.0, 1000.0).b.empty());
    EXPECT_TRUE(cheby2(2, 40.0, 600.0, 1000.0).b.empty());
}

TEST(SignalCheby2Test, invalid_fs_or_attenuation_returns_empty) {
    EXPECT_TRUE(cheby2(2, 40.0, 100.0, 0.0).b.empty());
    EXPECT_TRUE(cheby2(2, -1.0, 100.0, 1000.0).b.empty());
}

TEST(SignalCheby2Test, lowpass_second_order_matches_scipy_reference) {
    // scipy.signal.cheby2(2, 40.0, 0.25, fs=2.0, btype='low')
    const auto coeffs = cheby2(2, 40.0, 0.25, 2.0, FilterType::Lowpass);
    ASSERT_EQ(coeffs.b.size(), 3u);
    ASSERT_EQ(coeffs.a.size(), 3u);
    EXPECT_NEAR(coeffs.b[0], 0.01236943, 1e-6);
    EXPECT_NEAR(coeffs.b[1], -0.01209834, 1e-6);
    EXPECT_NEAR(coeffs.b[2], 0.01236943, 1e-6);
    EXPECT_NEAR(coeffs.a[0], 1.0, 1e-12);
    EXPECT_NEAR(coeffs.a[1], -1.83553964, 1e-6);
    EXPECT_NEAR(coeffs.a[2], 0.84818017, 1e-6);
}

TEST(SignalCheby2Test, highpass_second_order_matches_scipy_reference) {
    // scipy.signal.cheby2(2, 40.0, 0.25, fs=2.0, btype='high')
    const auto coeffs = cheby2(2, 40.0, 0.25, 2.0, FilterType::Highpass);
    ASSERT_EQ(coeffs.b.size(), 3u);
    ASSERT_EQ(coeffs.a.size(), 3u);
    EXPECT_NEAR(coeffs.b[0], 0.07925439, 1e-6);
    EXPECT_NEAR(coeffs.b[1], -0.13346167, 1e-6);
    EXPECT_NEAR(coeffs.b[2], 0.07925439, 1e-6);
    EXPECT_NEAR(coeffs.a[0], 1.0, 1e-12);
    EXPECT_NEAR(coeffs.a[1], 1.10637001, 1e-6);
    EXPECT_NEAR(coeffs.a[2], 0.39834045, 1e-6);
}

TEST(SignalCheby2Test, lowpass_dc_magnitude_exceeds_nyquist) {
    const auto coeffs = cheby2(4, 60.0, 100.0, 1000.0, FilterType::Lowpass);
    ASSERT_FALSE(coeffs.b.empty());
    ASSERT_FALSE(coeffs.a.empty());
    const double dc = freqz_mag_dc(coeffs.b, coeffs.a);
    const double nyq = freqz_mag_nyquist(coeffs.b, coeffs.a);
    EXPECT_GT(dc, nyq);
    EXPECT_GT(dc, 0.5);
    EXPECT_LT(nyq, 0.01);
}

TEST(SignalCheby2Test, highpass_nyquist_exceeds_dc) {
    const auto coeffs = cheby2(4, 60.0, 100.0, 1000.0, FilterType::Highpass);
    ASSERT_FALSE(coeffs.b.empty());
    const double dc = freqz_mag_dc(coeffs.b, coeffs.a);
    const double nyq = freqz_mag_nyquist(coeffs.b, coeffs.a);
    EXPECT_LT(dc, 0.01);
    EXPECT_GT(nyq, 0.5);
}

TEST(SignalCheby2Test, denominator_leading_coefficient_is_one) {
    for (const FilterType type : {FilterType::Lowpass, FilterType::Highpass}) {
        const auto coeffs = cheby2(3, 30.0, 250.0, 1000.0, type);
        ASSERT_FALSE(coeffs.a.empty());
        EXPECT_NEAR(coeffs.a[0], 1.0, 1e-12);
    }
}

TEST(SignalCheby2Test, filter_applies_designed_coefficients) {
    const auto coeffs = cheby2(2, 40.0, 0.25, 2.0);
    const std::vector<double> x{1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0};
    const auto y = filter(coeffs.b, coeffs.a, x);
    ASSERT_EQ(y.size(), x.size());
    for (double v : y) {
        EXPECT_TRUE(std::isfinite(v));
    }
    EXPECT_GT(std::abs(y.back()), 0.0);
}

TEST(SignalCheby2Test, fourth_order_lowpass_matches_scipy_reference) {
    // scipy.signal.cheby2(4, 60.0, 100.0, fs=1000.0, btype='low')
    const auto coeffs = cheby2(4, 60.0, 100.0, 1000.0, FilterType::Lowpass);
    ASSERT_EQ(coeffs.b.size(), 5u);
    ASSERT_EQ(coeffs.a.size(), 5u);
    EXPECT_NEAR(coeffs.b[0], 0.00150402, 1e-5);
    EXPECT_NEAR(coeffs.b[4], 0.00150402, 1e-5);
    EXPECT_NEAR(coeffs.a[1], -3.49859765, 1e-5);
    EXPECT_NEAR(coeffs.a[4], 0.60493786, 1e-5);
}

TEST(SignalCheby2Test, higher_attenuation_reduces_stopband_gain) {
    const auto mild = cheby2(2, 20.0, 0.25, 2.0, FilterType::Lowpass);
    const auto sharp = cheby2(2, 40.0, 0.25, 2.0, FilterType::Lowpass);
    const double nyq_mild = freqz_mag_nyquist(mild.b, mild.a);
    const double nyq_sharp = freqz_mag_nyquist(sharp.b, sharp.a);
    EXPECT_LT(nyq_sharp, nyq_mild);
}

// ---------------------------------------------------------------------------
// sosfilt(): second-order-sections IIR cascade via filter().
// ---------------------------------------------------------------------------

TEST(SignalSosfiltTest, empty_sos_returns_x_unchanged) {
    const std::vector<double> x{1.0, -2.0, 3.5, 0.0};
    const auto y = sosfilt({}, x);
    ASSERT_EQ(y.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_DOUBLE_EQ(y[i], x[i]);
    }
}

TEST(SignalSosfiltTest, empty_input_with_nonempty_sos_returns_empty) {
    const std::vector<std::array<double, 6>> sos{{1.0, 0.0, 0.0, 1.0, 0.0, 0.0}};
    EXPECT_TRUE(sosfilt(sos, {}).empty());
}

TEST(SignalSosfiltTest, empty_sos_and_empty_input_returns_empty) {
    EXPECT_TRUE(sosfilt({}, {}).empty());
}

TEST(SignalSosfiltTest, single_identity_section) {
    const std::vector<double> x{1.0, -2.0, 3.5, 0.0, -7.25};
    const std::vector<std::array<double, 6>> sos{{1.0, 0.0, 0.0, 1.0, 0.0, 0.0}};
    const auto y = sosfilt(sos, x);
    ASSERT_EQ(y.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_DOUBLE_EQ(y[i], x[i]);
    }
}

TEST(SignalSosfiltTest, single_section_matches_filter) {
    const std::vector<double> b{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
    const std::vector<double> a{1.0};
    const std::vector<double> x{0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const std::vector<std::array<double, 6>> sos{{b[0], b[1], b[2], a[0], 0.0, 0.0}};
    const auto y_sos = sosfilt(sos, x);
    const auto y_filter = filter(b, a, x);
    ASSERT_EQ(y_sos.size(), y_filter.size());
    for (size_t i = 0; i < y_filter.size(); ++i) {
        EXPECT_NEAR(y_sos[i], y_filter[i], 1e-12);
    }
}

TEST(SignalSosfiltTest, cascades_two_sections) {
    // Section 1: 3-tap moving average. Section 2: first-order IIR y = 0.5*x + 0.5*y[-1].
    const std::vector<std::array<double, 6>> sos{
        {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0, 0.0, 0.0},
        {0.5, 0.0, 0.0, 1.0, -0.5, 0.0},
    };
    const std::vector<double> x{0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const auto y = sosfilt(sos, x);

    const auto stage1 = filter({1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}, {1.0}, x);
    const auto expected = filter({0.5}, {1.0, -0.5}, stage1);
    ASSERT_EQ(y.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y[i], expected[i], 1e-12);
    }
}

TEST(SignalSosfiltTest, normalizes_a0_per_section) {
    // Same as filter_normalizes_when_a0_not_one, expressed as one SOS row scaled by 2.
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<std::array<double, 6>> sos{{2.0, -2.0, 0.0, 2.0, -1.0, 0.0}};
    const auto y = sosfilt(sos, x);
    const std::vector<double> expected{1.0, 1.5, 1.75, 1.875, 1.9375};
    ASSERT_EQ(y.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y[i], expected[i], 1e-12);
    }
}

TEST(SignalSosfiltTest, second_order_butterworth_matches_direct_form) {
    // scipy.signal.butter(2, 0.25, output='sos') — single 2nd-order Butterworth lowpass section.
    const std::vector<std::array<double, 6>> sos{
        {0.09763107, 0.19526215, 0.09763107, 1.0, -0.94280904, 0.33333333},
    };
    const std::vector<double> b{0.09763107, 0.19526215, 0.09763107};
    const std::vector<double> a{1.0, -0.94280904, 0.33333333};
    const std::vector<double> x{1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0};
    const auto y_sos = sosfilt(sos, x);
    const auto y_direct = filter(b, a, x);
    const std::vector<double> expected{
        0.09763107, 0.2873096, 0.2383344, -0.06632819, -0.14197961, 0.08351188, 0.12606229,
        -0.10424677,
    };
    ASSERT_EQ(y_sos.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y_sos[i], expected[i], 1e-6);
        EXPECT_NEAR(y_sos[i], y_direct[i], 1e-12);
    }
}

TEST(SignalSosfiltTest, fourth_order_butterworth_chain_matches_direct_form) {
    // scipy.signal.butter(4, 0.25, output='sos') — two cascaded 2nd-order Butterworth sections.
    const std::vector<std::array<double, 6>> sos{
        {0.01020948, 0.02041896, 0.01020948, 1.0, -0.85539793, 0.20971536},
        {1.0, 2.0, 1.0, 1.0, -1.11302985, 0.57406192},
    };
    const std::vector<double> x{0.0, 1.0, 0.5, -0.25, 0.75, -1.0, 0.3, 0.6, -0.4, 0.2,
                                0.9, -0.1, 0.4, -0.6, 0.8, 0.0};
    const auto y_sos = sosfilt(sos, x);
    const auto stage1 = filter({0.01020948, 0.02041896, 0.01020948},
                               {1.0, -0.85539793, 0.20971536}, x);
    const auto expected = filter({1.0, 2.0, 1.0}, {1.0, -1.11302985, 0.57406192}, stage1);
    ASSERT_EQ(y_sos.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(y_sos[i], expected[i], 1e-12);
    }
}

TEST(SignalSosfiltTest, butterworth_preserves_length) {
    const std::vector<std::array<double, 6>> sos{
        {0.09763107, 0.19526215, 0.09763107, 1.0, -0.94280904, 0.33333333},
    };
    std::vector<double> x(32, 0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(2.0 * M_PI * 0.05 * static_cast<double>(i));
    }
    const auto y = sosfilt(sos, x);
    EXPECT_EQ(y.size(), x.size());
    for (double v : y) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(SignalSosfiltTest, causal_sosfilt_differs_from_filtfilt_zero_phase) {
    // sosfilt is causal; filtfilt on the same direct-form coeffs removes group delay.
    const std::vector<std::array<double, 6>> sos{
        {0.09763107, 0.19526215, 0.09763107, 1.0, -0.94280904, 0.33333333},
    };
    const std::vector<double> b{0.09763107, 0.19526215, 0.09763107};
    const std::vector<double> a{1.0, -0.94280904, 0.33333333};
    std::vector<double> x(64, 0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = (i < 32) ? 0.0 : 1.0;
    }
    const auto y_causal = sosfilt(sos, x);
    const auto y_zero_phase = filtfilt(b, a, x);
    ASSERT_EQ(y_causal.size(), y_zero_phase.size());
    double max_abs_diff = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, std::abs(y_causal[i] - y_zero_phase[i]));
    }
    EXPECT_GT(max_abs_diff, 0.01);
    // At the step edge, filtfilt tracks the input sooner than the causal pass.
    EXPECT_GT(y_zero_phase[34], y_causal[34]);
    EXPECT_NEAR(y_zero_phase[50], 1.0, 0.05);
}

// ---------------------------------------------------------------------------
// filter_in_place() / filter(x, y span): scipy lfilter parity and sosfilt support.
// ---------------------------------------------------------------------------

TEST(SignalFilterInPlaceTest, fir_impulse_matches_filter) {
    const std::vector<double> b{0.25, 0.5, 0.25};
    const std::vector<double> a{1.0};
    std::vector<double> impulse(16, 0.0);
    impulse[0] = 1.0;
    const auto expected = filter(b, a, impulse);
    std::vector<double> y = impulse;
    filter_in_place(b, a, std::span<double>(y));
    ASSERT_EQ(y.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y[i], expected[i], 1e-12);
    }
}

TEST(SignalFilterInPlaceTest, iir_matches_scipy_lfilter_reference) {
    // scipy.signal.lfilter(b, a, [1,0,-1,0,1,0,-1,0])
    const std::vector<double> b{0.09763107, 0.19526215, 0.09763107};
    const std::vector<double> a{1.0, -0.94280904, 0.33333333};
    const std::vector<double> x{1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0};
    const std::vector<double> expected{
        0.09763107, 0.2873096, 0.2383344, -0.06632819,
        -0.14197961, 0.08351188, 0.12606229, -0.10424677,
    };
    std::vector<double> y = x;
    filter_in_place(b, a, std::span<double>(y));
    ASSERT_EQ(y.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y[i], expected[i], 1e-6);
    }
}

TEST(SignalFilterInPlaceTest, output_span_matches_filter) {
    const std::vector<double> b{0.5};
    const std::vector<double> a{1.0, -0.5};
    const std::vector<double> x{0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<double> y(x.size(), -1.0);
    filter(b, a, std::span<const double>(x), std::span<double>(y));
    const auto expected = filter(b, a, x);
    ASSERT_EQ(y.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y[i], expected[i], 1e-12);
    }
}

TEST(SignalFilterInPlaceTest, normalizes_a0_in_place) {
    const std::vector<double> b{2.0, -2.0, 0.0};
    const std::vector<double> a{2.0, -1.0, 0.0};
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = x;
    filter_in_place(b, a, std::span<double>(y));
    const std::vector<double> expected{1.0, 1.5, 1.75, 1.875, 1.9375};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y[i], expected[i], 1e-12);
    }
}

TEST(SignalSosfiltScipyTest, second_order_impulse_matches_scipy_sosfilt) {
    // scipy.signal.sosfilt(scipy.signal.butter(2, 0.25, output='sos'), unit impulse)
    const std::vector<std::array<double, 6>> sos{
        {0.09763107, 0.19526215, 0.09763107, 1.0, -0.94280904, 0.33333333},
    };
    std::vector<double> impulse(16, 0.0);
    impulse[0] = 1.0;
    const auto y = sosfilt(sos, impulse);
    const std::vector<double> expected{
        0.09763107, 0.28730961, 0.33596547, 0.22098142,
        0.09635479, 0.01718369, -0.01591732, -0.02073489,
    };
    ASSERT_GE(y.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y[i], expected[i], 1e-6);
    }
}

TEST(SignalSosfiltScipyTest, fourth_order_step_matches_scipy_sosfilt) {
    // scipy.signal.sosfilt(scipy.signal.butter(4, 0.25, output='sos'), step at n=4)
    const std::vector<std::array<double, 6>> sos{
        {0.01020948, 0.02041896, 0.01020948, 1.0, -0.85539793, 0.20971536},
        {1.0, 2.0, 1.0, 1.0, -1.11302985, 0.57406192},
    };
    std::vector<double> step(20, 0.0);
    for (size_t i = 4; i < step.size(); ++i) {
        step[i] = 1.0;
    }
    const auto y = sosfilt(sos, step);
    const std::vector<double> expected_tail{
        0.01020948, 0.07114403, 0.23462394, 0.49888283, 0.78840471, 1.01069152,
        1.11744398, 1.11965569, 1.06488676, 1.00381688, 0.9674269, 0.96185989,
        0.97641985, 0.99573232, 1.00882129, 1.01228073,
    };
    ASSERT_EQ(y.size(), step.size());
    for (size_t i = 0; i < expected_tail.size(); ++i) {
        EXPECT_NEAR(y[i + 4], expected_tail[i], 1e-5);
    }
}

TEST(SignalSosfiltScipyTest, mixed_signal_matches_scipy_sosfilt) {
    // scipy.signal.sosfilt(butter(4,0.25,'sos'), alternating sine-like pattern)
    const std::vector<std::array<double, 6>> sos{
        {0.01020948, 0.02041896, 0.01020948, 1.0, -0.85539793, 0.20971536},
        {1.0, 2.0, 1.0, 1.0, -1.11302985, 0.57406192},
    };
    const std::vector<double> x{1.0, -0.5, 0.25, -0.125, 0.0625, -0.03125,
                                0.015625, -0.0078125, 0.00390625, -0.001953125};
    const auto y = sosfilt(sos, x);
    const std::vector<double> expected{
        0.010209480791203138, 0.05582980844883029, 0.13556500308587335,
        0.19647639562280456, 0.19128367910091504, 0.12664496934309172,
        0.04342997783811017, -0.01950328169628366, -0.04501729257481829,
        -0.038561234710196936,
    };
    ASSERT_EQ(y.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y[i], expected[i], 1e-5);
    }
}

TEST(SignalSosfiltScipyTest, input_vector_unchanged) {
    const std::vector<std::array<double, 6>> sos{
        {0.09763107, 0.19526215, 0.09763107, 1.0, -0.94280904, 0.33333333},
    };
    const std::vector<double> x_orig{1.0, 0.0, -1.0, 0.5, -0.25};
    const std::vector<double> x = x_orig;
    const auto y = sosfilt(sos, x);
    ASSERT_EQ(x, x_orig);
    ASSERT_EQ(y.size(), x.size());
    EXPECT_NE(y, x);
}

TEST(SignalSosfiltScipyTest, single_section_via_filter_in_place_matches_sosfilt) {
    const std::vector<std::array<double, 6>> sos{
        {0.09763107, 0.19526215, 0.09763107, 1.0, -0.94280904, 0.33333333},
    };
    const std::vector<double> b{0.09763107, 0.19526215, 0.09763107};
    const std::vector<double> a{1.0, -0.94280904, 0.33333333};
    const std::vector<double> x{1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0};
    std::vector<double> y = x;
    filter_in_place(b, a, std::span<double>(y));
    const auto y_sos = sosfilt(sos, x);
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(y[i], y_sos[i], 1e-12);
    }
}

// ---- cheby1 IIR design ----

static double iir_dc_gain(const IirCoeffs& c) {
    if (c.b.empty() || c.a.empty()) return 0.0;
    double num = 0.0, den = 0.0;
    for (double v : c.b) num += v;
    for (double v : c.a) den += v;
    return (std::abs(den) > 1e-15) ? num / den : 0.0;
}

TEST(SignalCheby1, InvalidArgsReturnEmpty) {
    EXPECT_TRUE(cheby1(0, 1.0, 1000.0, 8000.0).b.empty());
    EXPECT_TRUE(cheby1(2, -1.0, 1000.0, 8000.0).b.empty());
    EXPECT_TRUE(cheby1(2, 1.0, 0.0, 8000.0).b.empty());
    EXPECT_TRUE(cheby1(2, 1.0, 5000.0, 8000.0).b.empty());
}

TEST(SignalCheby1, LowpassSecondOrderNormalized) {
    const auto c = cheby1(2, 1.0, 1000.0, 8000.0, FilterType::Lowpass);
    ASSERT_EQ(c.a.size(), 3u);
    ASSERT_EQ(c.b.size(), 3u);
    EXPECT_NEAR(c.a[0], 1.0, 1e-12);
    EXPECT_GT(std::abs(iir_dc_gain(c)), 0.5);
}

TEST(SignalCheby1, HighpassAttenuatesDC) {
    const auto c = cheby1(2, 1.0, 1000.0, 8000.0, FilterType::Highpass);
    ASSERT_FALSE(c.b.empty());
    EXPECT_LT(std::abs(iir_dc_gain(c)), 0.1);
}

TEST(SignalCheby1, FilterProducesFiniteOutput) {
    const auto c = cheby1(4, 0.5, 500.0, 4000.0);
    std::vector<double> x(64);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(2.0 * M_PI * 50.0 * static_cast<double>(i) / 4000.0);
    }
    const auto y = filter(c.b, c.a, x);
    ASSERT_EQ(y.size(), x.size());
    for (double v : y) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(SignalCheby1, HigherOrderMoreCoefficients) {
    const auto c2 = cheby1(2, 1.0, 1000.0, 8000.0);
    const auto c4 = cheby1(4, 1.0, 1000.0, 8000.0);
    EXPECT_LT(c2.b.size(), c4.b.size());
    EXPECT_LT(c2.a.size(), c4.a.size());
}

TEST(SignalCheby1, LowpassReducesHighFrequencyEnergy) {
    const auto c = cheby1(4, 1.0, 200.0, 4000.0);
    std::vector<double> x(256);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(2.0 * M_PI * 1500.0 * static_cast<double>(i) / 4000.0);
    }
    const auto y = filter(c.b, c.a, x);
    double in_sq = 0.0, out_sq = 0.0;
    for (size_t i = 32; i < x.size(); ++i) {
        in_sq += x[i] * x[i];
        out_sq += y[i] * y[i];
    }
    const double in_rms = std::sqrt(in_sq / (x.size() - 32));
    const double out_rms = std::sqrt(out_sq / (y.size() - 32));
    EXPECT_LT(out_rms, 0.5 * in_rms);
}

TEST(SignalCheby1, SosfiltCompatible) {
    const auto c = cheby1(2, 1.0, 1000.0, 8000.0);
    const std::vector<std::array<double, 6>> sos{{c.b[0], c.b[1], c.b[2], c.a[0], c.a[1], c.a[2]}};
    const std::vector<double> x{1.0, 0.0, -1.0, 0.5, -0.25};
    const auto y1 = filter(c.b, c.a, x);
    const auto y2 = sosfilt(sos, x);
    ASSERT_EQ(y1.size(), y2.size());
    for (size_t i = 0; i < y1.size(); ++i) {
        EXPECT_NEAR(y1[i], y2[i], 1e-10);
    }
}

TEST(SignalCheby1, RippleParameterAffectsGain) {
    const auto mild = cheby1(3, 0.5, 1000.0, 8000.0);
    const auto sharp = cheby1(3, 3.0, 1000.0, 8000.0);
    EXPECT_NE(iir_dc_gain(mild), iir_dc_gain(sharp));
}

// ---------------------------------------------------------------------------
// Wave 218: FFT-accelerated convolve / correlate
// ---------------------------------------------------------------------------

namespace {

std::vector<double> reference_convolve_direct(const std::vector<double>& a,
                                              const std::vector<double>& b) {
    std::vector<double> result(a.size() + b.size() - 1, 0.0);
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            result[i + j] += a[i] * b[j];
        }
    }
    return result;
}

} // namespace

TEST(SignalConvolveTest, SmallDirectPath) {
    const std::vector<double> a{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> b{0.5, -0.25};
    const auto out = convolve(a, b);
    const std::vector<double> expected{0.5, 0.75, 1.0, 1.25, -1.0};
    ASSERT_EQ(out.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(out[i], expected[i]);
    }
}

TEST(SignalConvolveTest, LargeFftPath) {
    constexpr size_t n = 256;
    std::vector<double> a(n), b(32);
    for (size_t i = 0; i < n; ++i) {
        a[i] = std::sin(2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n));
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = 1.0 / static_cast<double>(b.size());
    }
    const auto out = convolve(a, b);
    const auto ref = reference_convolve_direct(a, b);
    ASSERT_EQ(out.size(), ref.size());
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], ref[i], 1e-10) << "mismatch at i=" << i;
    }
}

TEST(SignalConvolveTest, EmptyInput) {
    const std::vector<double> empty;
    const std::vector<double> b{1.0, 2.0};
    const auto ab = convolve(empty, b);
    const auto ba = convolve(b, empty);
    EXPECT_EQ(ab.size(), b.size() - 1);
    EXPECT_EQ(ba.size(), b.size() - 1);
    for (double v : ab) {
        EXPECT_DOUBLE_EQ(v, 0.0);
    }
    for (double v : ba) {
        EXPECT_DOUBLE_EQ(v, 0.0);
    }
}

TEST(SignalConvolveTest, SingleElement) {
    const std::vector<double> a{3.0};
    const std::vector<double> b{4.0};
    const auto out = convolve(a, b);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_DOUBLE_EQ(out[0], 12.0);
}

TEST(SignalConvolveTest, IdentityKernel) {
    const std::vector<double> x{1.0, -2.0, 3.5, 0.25};
    const std::vector<double> impulse{1.0};
    const auto out = convolve(x, impulse);
    ASSERT_EQ(out.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_DOUBLE_EQ(out[i], x[i]);
    }
}

TEST(SignalConvolveTest, FftMatchesDirectAtMediumSize) {
    constexpr size_t n = 128;
    std::vector<double> a(n), b(n / 2);
    for (size_t i = 0; i < n; ++i) {
        a[i] = std::cos(2.0 * M_PI * 7.0 * static_cast<double>(i) / static_cast<double>(n));
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = std::exp(-0.05 * static_cast<double>(i));
    }
    const auto out = convolve(a, b);
    const auto ref = reference_convolve_direct(a, b);
    ASSERT_EQ(out.size(), ref.size());
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], ref[i], 1e-10) << "FFT vs direct mismatch at i=" << i;
    }
}

TEST(SignalConvolveTest, CorrelateReusesConvolve) {
    const std::vector<double> a{1.0, 2.0, 3.0};
    const std::vector<double> b{4.0, 5.0};
    const auto r = correlate(a, b);
    std::vector<double> rev_b(b.rbegin(), b.rend());
    const auto c = convolve(a, rev_b);
    ASSERT_EQ(r.size(), c.size());
    for (size_t i = 0; i < r.size(); ++i) {
        EXPECT_DOUBLE_EQ(r[i], c[i]);
    }
}

TEST(SignalConvolveTest, LargeCorrelateFftPath) {
    constexpr size_t n = 512;
    std::vector<double> a(n), b(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = std::sin(2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n));
        b[i] = std::cos(2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n));
    }
    const auto out = correlate(a, b);
    std::vector<double> rev_b(b.rbegin(), b.rend());
    const auto ref = reference_convolve_direct(a, rev_b);
    ASSERT_EQ(out.size(), ref.size());
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], ref[i], 1e-10) << "correlate mismatch at i=" << i;
    }
}
