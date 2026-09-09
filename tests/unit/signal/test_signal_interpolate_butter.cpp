// Regression tests for two audited signal defects.
//
//  * interpolate()'s frequency-domain fast path periodises the input spectrum as
//    spec_out[k] = spec_in[k % in_fft] with in_fft = next_pow2(n) and
//    out_fft = next_pow2(n*p). That zero-stuffs by out_fft/in_fft, which equals
//    p only when p is a power of two. The path was taken for every p >= 8, so
//    p = 9, 10, 11, ... produced a differently-stuffed, truncated, mis-scaled
//    signal: a 32-sample ramp interpolated by 8 peaks at 34.1 (correct) but by
//    9 at 10.4 and by 10 at 12.8.
//  * butterworth() was byte-for-byte ms::lowpass -- an ideal brick-wall FFT mask
//    with no filter order, no analog prototype, and none of the maximally-flat
//    magnitude or finite rolloff the name denotes.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "ms/signal/signal.hpp"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

std::vector<double> ramp(std::size_t n) {
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i);
    }
    return x;
}

double peak_abs(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) {
        m = std::max(m, std::abs(x));
    }
    return m;
}

// Amplitude of a steady sinusoid after filtering, measured away from the edges.
double steady_amplitude(const std::vector<double>& y) {
    double m = 0.0;
    for (std::size_t i = y.size() / 4; i < 3 * y.size() / 4; ++i) {
        m = std::max(m, std::abs(y[i]));
    }
    return m;
}

} // namespace

TEST(SignalInterpolate, PeakScalesConsistentlyAcrossFactors) {
    // The output of a band-limited interpolation of a ramp tracks the input's
    // own range. Before the fix p = 9 and p = 10 collapsed the peak by roughly
    // n*p/out_fft while p = 8 was correct, so the peak jumped around wildly.
    const auto x = ramp(32);
    const double in_peak = peak_abs(x);
    for (int p = 3; p <= 12; ++p) {
        const auto y = ms::interpolate(x, p);
        ASSERT_EQ(y.size(), x.size() * static_cast<std::size_t>(p)) << "p = " << p;
        const double out_peak = peak_abs(y);
        // Allow generous slack for the low-pass transition, but the pre-fix
        // values (10.4 and 12.8 against an input peak of 31) are far outside it.
        EXPECT_GT(out_peak, 0.7 * in_peak) << "p = " << p;
        EXPECT_LT(out_peak, 1.5 * in_peak) << "p = " << p;
    }
}

TEST(SignalInterpolate, ReproducesABandLimitedInputAtEveryPthSample) {
    // A sinusoid well inside the passband must survive interpolation, so the
    // samples at multiples of p come back close to the original.
    std::vector<double> x(32);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(2.0 * M_PI * 3.0 * static_cast<double>(i) / 32.0);
    }
    for (int p : {3, 4, 8, 9, 10, 11, 12}) {
        const auto y = ms::interpolate(x, p);
        ASSERT_EQ(y.size(), x.size() * static_cast<std::size_t>(p));
        double worst = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            worst = std::max(worst, std::abs(y[i * static_cast<std::size_t>(p)] - x[i]));
        }
        EXPECT_LT(worst, 0.1) << "p = " << p;
    }
}

TEST(SignalInterpolate, NonPowerOfTwoFactorsMatchTheirNeighbours) {
    // p = 8 (power of two, fast path) and p = 9 (reference path) must produce
    // comparably scaled output. The pre-fix ratio was about 3x.
    const auto x = ramp(32);
    const double p8 = peak_abs(ms::interpolate(x, 8));
    const double p9 = peak_abs(ms::interpolate(x, 9));
    EXPECT_GT(p9, 0.7 * p8);
    EXPECT_LT(p9, 1.4 * p8);
}

TEST(SignalButterworth, HasAMonotoneMaximallyFlatResponse) {
    // filtfilt applies the filter forwards and backwards, so the measured gain
    // is |H|^2 = 1 / (1 + (f/fc)^(2n)) with n = 4.
    const double fs = 1.0;
    const double fc = 0.1;
    const std::size_t N = 4096;
    double prev = 2.0;
    for (double f : {0.02, 0.05, 0.10, 0.20, 0.30}) {
        std::vector<double> x(N);
        for (std::size_t i = 0; i < N; ++i) {
            x[i] = std::sin(2.0 * M_PI * f * static_cast<double>(i));
        }
        const double gain = steady_amplitude(ms::butterworth(x, fc, fs));
        const double theory = 1.0 / (1.0 + std::pow(f / fc, 8.0));
        EXPECT_NEAR(gain, theory, 0.05) << "f = " << f;
        EXPECT_LT(gain, prev) << "response must be monotone decreasing at f = " << f;
        prev = gain;
    }
}

TEST(SignalButterworth, IsHalfPowerAtTheCutoff) {
    // The defining property: |H(fc)|^2 = 1/2 for any order, so filtfilt gives
    // 1/2 as the measured gain. A brick-wall mask gives ~1 instead.
    const std::size_t N = 4096;
    const double fc = 0.1;
    std::vector<double> x(N);
    for (std::size_t i = 0; i < N; ++i) {
        x[i] = std::sin(2.0 * M_PI * fc * static_cast<double>(i));
    }
    EXPECT_NEAR(steady_amplitude(ms::butterworth(x, fc, 1.0)), 0.5, 0.05);
}

TEST(SignalButterworth, DiffersFromTheBrickWallLowpass) {
    // An existing test asserted butterworth == lowpass to 1e-12, which blessed
    // the aliasing. Well above the cutoff a brick wall gives essentially zero
    // while a finite-order Butterworth leaves a small but distinguishable
    // residue, and just below the cutoff the brick wall passes unity while the
    // Butterworth has already begun to roll off.
    const std::size_t N = 2048;
    std::vector<double> x(N);
    for (std::size_t i = 0; i < N; ++i) {
        x[i] = std::sin(2.0 * M_PI * 0.09 * static_cast<double>(i));
    }
    const double b = steady_amplitude(ms::butterworth(x, 0.1, 1.0));
    const double l = steady_amplitude(ms::lowpass(x, 0.1, 1.0));
    EXPECT_GT(std::abs(b - l), 1e-3) << "butterworth is still aliased to lowpass";
}

TEST(SignalButterworth, HigherOrderRollsOffFaster) {
    const std::size_t N = 4096;
    std::vector<double> x(N);
    for (std::size_t i = 0; i < N; ++i) {
        x[i] = std::sin(2.0 * M_PI * 0.2 * static_cast<double>(i));
    }
    const double g2 = steady_amplitude(ms::butterworth(x, 0.1, 1.0, 2));
    const double g6 = steady_amplitude(ms::butterworth(x, 0.1, 1.0, 6));
    EXPECT_LT(g6, g2);
}

TEST(SignalButterworth, DegenerateInputIsReturnedUnchanged) {
    const std::vector<double> empty;
    EXPECT_TRUE(ms::butterworth(empty, 0.1, 1.0).empty());
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    EXPECT_EQ(ms::butterworth(x, 0.0, 1.0).size(), x.size());   // cutoff <= 0
    EXPECT_EQ(ms::butterworth(x, 5.0, 1.0).size(), x.size());   // cutoff >= fs/2
    EXPECT_EQ(ms::butterworth(x, 0.1, 1.0, 0).size(), x.size()); // order < 1
}
