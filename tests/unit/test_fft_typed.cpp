#include <complex>
#include <gtest/gtest.h>
#include <cmath>

#include "ms/fft/fft.hpp"

using namespace ms;

namespace {

double energy_time(const std::vector<double>& x) {
    double sum = 0.0;
    for (double v : x) {
        sum += v * v;
    }
    return sum;
}

double energy_freq(const std::vector<std::complex<double>>& X) {
    double sum = 0.0;
    for (const auto& v : X) {
        sum += std::norm(v);
    }
    return sum;
}

} // namespace

TEST(FftTypedTest, roundtrip_float) {
    // Use power-of-2 size to avoid zero-padding artifacts
    const std::vector<double> x{1.0, -1.0, 0.5, 2.0, -0.25, 0.75, 1.5, -0.5};
    const auto spec = fft(x);
    ASSERT_TRUE(spec.has_value());
    const auto back = ifft(*spec);
    ASSERT_TRUE(back.has_value());
    ASSERT_EQ(back->size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR((*back)[i], x[i], 1e-6);
    }
}

TEST(FftTypedTest, parseval_theorem) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const auto spec = fft(x);
    ASSERT_TRUE(spec.has_value());
    const double time_energy = energy_time(x);
    const double freq_energy = energy_freq(*spec) / static_cast<double>(x.size());
    EXPECT_NEAR(time_energy, freq_energy, 1e-5);
}

TEST(FftTypedTest, complex_spectrum_roundtrip) {
    // Test fft->ifft roundtrip using a real power-of-2 signal
    const std::vector<double> x{3.0, -1.0, 2.0, 0.5};
    const auto spec = fft(x);
    ASSERT_TRUE(spec.has_value());
    const auto back = ifft(*spec);
    ASSERT_TRUE(back.has_value());
    ASSERT_EQ(back->size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR((*back)[i], x[i], 1e-5);
    }
}
