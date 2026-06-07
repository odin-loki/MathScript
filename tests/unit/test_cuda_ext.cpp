#include <gtest/gtest.h>
#include "ms/cuda/elementwise.hpp"
#include "ms/cuda/fft.hpp"
#include <cmath>

using namespace ms;

TEST(CudaFftTest, returns_padded_spectrum) {
    std::vector<double> signal{1, 2, 3, 4};
    auto spectrum = cuda::fft(signal).value();
    EXPECT_EQ(spectrum.size(), 4);
    EXPECT_GT(std::abs(spectrum[0]), 0.0);
}

TEST(CudaFftTest, dc_component) {
    std::vector<double> signal{2, 2, 2, 2};
    auto spectrum = cuda::fft(signal).value();
    EXPECT_NEAR(spectrum[0].real(), 8.0, 1e-3);
}

TEST(CudaElementwiseTest, add_inplace) {
    std::vector<double> a{1, 2, 3};
    std::vector<double> b{4, 5, 6};
    cuda::add_inplace(a, b);
    EXPECT_DOUBLE_EQ(a[0], 5.0);
    EXPECT_DOUBLE_EQ(a[2], 9.0);
    cuda::fill(a, 0.0);
    EXPECT_DOUBLE_EQ(a[1], 0.0);
}
