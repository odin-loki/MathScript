// MathScript Signal Window Reference Tests
// Placeholder for signal window basic tests (properties covered in test_signal_windows_adv4.cpp)

#include <gtest/gtest.h>
#include "ms/signal/signal.hpp"
using namespace ms;

TEST(SignalWindowsRef, Hamming_Size_And_Bounds) {
    auto w = hamming(32);
    EXPECT_EQ(w.size(), 32u);
    for (double v : w) { EXPECT_GE(v, 0.0); EXPECT_LE(v, 1.0 + 1e-10); }
}

TEST(SignalWindowsRef, Hanning_Size_And_Bounds) {
    auto w = hanning(32);
    EXPECT_EQ(w.size(), 32u);
    for (double v : w) { EXPECT_GE(v, -1e-10); EXPECT_LE(v, 1.0 + 1e-10); }
}

TEST(SignalWindowsRef, Blackman_Size_And_Bounds) {
    auto w = blackman(32);
    EXPECT_EQ(w.size(), 32u);
    for (double v : w) { EXPECT_GE(v, -1e-10); EXPECT_LE(v, 1.0 + 1e-10); }
}

TEST(SignalWindowsRef, Parzen_Size) {
    auto w = parzen(32);
    EXPECT_EQ(w.size(), 32u);
}

TEST(SignalWindowsRef, Triangular_Size) {
    auto w = triangular(32);
    EXPECT_EQ(w.size(), 32u);
}
