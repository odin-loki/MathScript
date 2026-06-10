#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include "ms/fft/fft.hpp"

using namespace ms;

namespace {

double signal_energy(const std::vector<double>& x) {
    double sum = 0.0;
    for (double v : x) {
        sum += v * v;
    }
    return sum;
}

double spectrum_energy(const std::vector<std::complex<double>>& X) {
    double sum = 0.0;
    for (const auto& v : X) {
        sum += std::norm(v);
    }
    return sum;
}

} // namespace

TEST(NumericalFft, Impulse_AllOnes) {
    const std::vector<double> x{1.0, 0.0, 0.0, 0.0};
    const auto X = fft(x).value();
    ASSERT_EQ(X.size(), 4u);
    for (const auto& v : X) {
        EXPECT_NEAR(v.real(), 1.0, 1e-13);
        EXPECT_NEAR(v.imag(), 0.0, 1e-13);
    }
}

TEST(NumericalFft, Constant_DeltaAtZero) {
    const std::vector<double> x{1.0, 1.0, 1.0, 1.0};
    const auto X = fft(x).value();
    ASSERT_EQ(X.size(), 4u);
    EXPECT_NEAR(X[0].real(), 4.0, 1e-13);
    EXPECT_NEAR(X[0].imag(), 0.0, 1e-13);
    for (std::size_t i = 1; i < X.size(); ++i) {
        EXPECT_NEAR(X[i].real(), 0.0, 1e-13);
        EXPECT_NEAR(X[i].imag(), 0.0, 1e-13);
    }
}

TEST(NumericalFft, Cosine_DeltaBins) {
    constexpr std::size_t N = 8;
    constexpr int k = 2;
    std::vector<double> x(N);
    for (std::size_t n = 0; n < N; ++n) {
        x[n] = std::cos(2.0 * std::numbers::pi * static_cast<double>(k) *
                        static_cast<double>(n) / static_cast<double>(N));
    }

    const auto X = fft(x).value();
    ASSERT_EQ(X.size(), N);

    for (std::size_t i = 0; i < N; ++i) {
        if (i == static_cast<std::size_t>(k) || i == N - static_cast<std::size_t>(k)) {
            EXPECT_NEAR(std::abs(X[i]), static_cast<double>(N) / 2.0, 1e-11);
        } else {
            EXPECT_NEAR(std::abs(X[i]), 0.0, 1e-11);
        }
    }
}

TEST(NumericalFft, RoundTrip_1e13) {
    const std::vector<double> x{1.0, -2.0, 3.5, -0.25, 7.0, 0.0, -1.5, 2.0};
    const auto back = ifft(fft(x).value()).value();
    ASSERT_EQ(back.size(), x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(back[i], x[i], 1e-13);
    }
}

TEST(NumericalFft, Parseval) {
    const std::vector<double> x{0.5, -1.25, 2.0, 0.0, -0.75, 3.0, 1.0, -2.5};
    const auto X = fft(x).value();
    const double time_energy = signal_energy(x);
    const double freq_energy = spectrum_energy(X) / static_cast<double>(x.size());
    EXPECT_NEAR(time_energy, freq_energy, 1e-12);
}
