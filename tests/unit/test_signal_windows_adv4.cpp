// MathScript Advanced Signal Window Tests (Wave 54)
// Parzen, Hanning, Hamming, Blackman, Triangular properties

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#include "ms/signal/signal.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// Parzen window
// ---------------------------------------------------------------------------

TEST(SignalWindows, Parzen_Size) {
    auto w = parzen(64);
    EXPECT_EQ(w.size(), 64u);
}

TEST(SignalWindows, Parzen_NonNeg) {
    auto w = parzen(32);
    for (double v : w) EXPECT_GE(v, -1e-10) << "Parzen should be non-negative";
}

TEST(SignalWindows, Parzen_MaxAtCenter) {
    auto w = parzen(64);
    double mx = *std::max_element(w.begin(), w.end());
    size_t center = 31; // 0-indexed center of 64-element window
    EXPECT_NEAR(w[center], mx, mx * 0.05) << "Parzen max should be near center";
}

TEST(SignalWindows, Parzen_AllFinite) {
    auto w = parzen(128);
    for (double v : w) EXPECT_TRUE(std::isfinite(v));
}

TEST(SignalWindows, Parzen_NearlySymmetric) {
    auto w = parzen(64);
    size_t n = w.size();
    double max_val = *std::max_element(w.begin(), w.end());
    for (size_t i = 0; i < n / 2; ++i)
        EXPECT_NEAR(w[i], w[n - 1 - i], max_val * 0.05)
            << "Parzen not symmetric at i=" << i;
}

// ---------------------------------------------------------------------------
// Hanning window
// ---------------------------------------------------------------------------

TEST(SignalWindows, Hanning_Size) {
    auto w = hanning(64);
    EXPECT_EQ(w.size(), 64u);
}

TEST(SignalWindows, Hanning_AllNonNeg) {
    auto w = hanning(64);
    for (double v : w) EXPECT_GE(v, -1e-10);
}

TEST(SignalWindows, Hanning_AllFinite) {
    auto w = hanning(128);
    for (double v : w) EXPECT_TRUE(std::isfinite(v));
}

TEST(SignalWindows, Hanning_MaxNearOne) {
    // Hanning window peak ≤ 1
    auto w = hanning(64);
    double mx = *std::max_element(w.begin(), w.end());
    EXPECT_LE(mx, 1.0 + 1e-10);
    EXPECT_GT(mx, 0.9) << "Hanning peak should be near 1";
}

TEST(SignalWindows, Hanning_NearlySymmetric) {
    auto w = hanning(64);
    size_t n = w.size();
    double max_val = *std::max_element(w.begin(), w.end());
    for (size_t i = 0; i < n / 2; ++i)
        EXPECT_NEAR(w[i], w[n - 1 - i], max_val * 0.05);
}

// ---------------------------------------------------------------------------
// Hamming window
// ---------------------------------------------------------------------------

TEST(SignalWindows, Hamming_Size) {
    auto w = hamming(64);
    EXPECT_EQ(w.size(), 64u);
}

TEST(SignalWindows, Hamming_AllFinite_NonNeg) {
    auto w = hamming(64);
    for (double v : w) {
        EXPECT_TRUE(std::isfinite(v));
        EXPECT_GE(v, -1e-10);
    }
}

TEST(SignalWindows, Hamming_MaxNearOne) {
    auto w = hamming(64);
    double mx = *std::max_element(w.begin(), w.end());
    EXPECT_GT(mx, 0.9);
    EXPECT_LE(mx, 1.0 + 1e-10);
}

TEST(SignalWindows, Hamming_MinAboveZero) {
    // Hamming window minimum is 0.08 (1 - 0.92)
    auto w = hamming(64);
    double mn = *std::min_element(w.begin(), w.end());
    EXPECT_GT(mn, 0.0) << "Hamming minimum should be above 0 (unlike Hanning)";
}

TEST(SignalWindows, Hamming_NearlySymmetric) {
    auto w = hamming(64);
    size_t n = w.size();
    double max_val = *std::max_element(w.begin(), w.end());
    for (size_t i = 0; i < n / 2; ++i)
        EXPECT_NEAR(w[i], w[n - 1 - i], max_val * 0.05);
}

// ---------------------------------------------------------------------------
// Blackman window
// ---------------------------------------------------------------------------

TEST(SignalWindows, Blackman_Size) {
    auto w = blackman(64);
    EXPECT_EQ(w.size(), 64u);
}

TEST(SignalWindows, Blackman_AllFinite_And_Bounded) {
    auto w = blackman(64);
    for (double v : w) {
        EXPECT_TRUE(std::isfinite(v));
        EXPECT_GE(v, -1e-10);
        EXPECT_LE(v, 1.0 + 1e-10);
    }
}

TEST(SignalWindows, Blackman_MaxNearOne) {
    auto w = blackman(64);
    double mx = *std::max_element(w.begin(), w.end());
    EXPECT_GT(mx, 0.9);
}

TEST(SignalWindows, Blackman_NearlySymmetric) {
    auto w = blackman(64);
    size_t n = w.size();
    double max_val = *std::max_element(w.begin(), w.end());
    for (size_t i = 0; i < n / 2; ++i)
        EXPECT_NEAR(w[i], w[n - 1 - i], max_val * 0.05);
}

// ---------------------------------------------------------------------------
// Windows comparison: energy ordering
// ---------------------------------------------------------------------------

TEST(SignalWindows, WindowEnergy_Blackman_Less_Than_Hamming) {
    // Blackman has narrower main lobe → less total energy than Hamming for same N
    size_t N = 64;
    auto bk = blackman(N);
    auto hm = hamming(N);
    double e_bk = 0.0, e_hm = 0.0;
    for (size_t i = 0; i < N; ++i) {
        e_bk += bk[i] * bk[i];
        e_hm += hm[i] * hm[i];
    }
    EXPECT_LT(e_bk, e_hm) << "Blackman should have less energy than Hamming";
}

TEST(SignalWindows, WindowEnergy_AllPositive) {
    for (size_t N : {32u, 64u, 128u}) {
        double e_parzen  = 0.0, e_hanning = 0.0, e_hamming = 0.0, e_blackman = 0.0;
        auto wp = parzen(N);
        auto wh = hanning(N);
        auto whm = hamming(N);
        auto wb = blackman(N);
        for (size_t i = 0; i < N; ++i) {
            e_parzen  += wp[i] * wp[i];
            e_hanning += wh[i] * wh[i];
            e_hamming += whm[i] * whm[i];
            e_blackman += wb[i] * wb[i];
        }
        EXPECT_GT(e_parzen, 0.0);
        EXPECT_GT(e_hanning, 0.0);
        EXPECT_GT(e_hamming, 0.0);
        EXPECT_GT(e_blackman, 0.0);
    }
}
