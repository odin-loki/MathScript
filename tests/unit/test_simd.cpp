#include <gtest/gtest.h>
#include "ms/simd/simd.hpp"
#include <cmath>
#include <numeric>
#include <vector>

using namespace ms::simd;

TEST(SimdTest, dispatch_detects_isa) {
    const auto info = dispatch_info();
    EXPECT_TRUE(info.isa.sse2);
}

TEST(SimdTest, add_matches_scalar) {
    std::vector<double> a(16);
    std::vector<double> b(16);
    std::vector<double> out(16);
    std::vector<double> ref(16);
    std::iota(a.begin(), a.end(), 0.0);
    std::iota(b.begin(), b.end(), 1.0);
    for (size_t i = 0; i < a.size(); ++i) {
        ref[i] = a[i] + b[i];
    }
    add(a, b, out);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], ref[i], 1e-12);
    }
}

TEST(SimdTest, dot_and_gemv) {
    std::vector<double> a{1, 2, 3, 4};
    std::vector<double> b{4, 3, 2, 1};
    EXPECT_NEAR(dot(a, b), 20.0, 1e-12);

    std::vector<double> A{1, 0, 0, 1};
    auto y = gemv(A, 2, 2, b);
    ASSERT_EQ(y.size(), 2);
    EXPECT_NEAR(y[0], 4.0, 1e-12);
    EXPECT_NEAR(y[1], 3.0, 1e-12);
}

TEST(SimdTest, axpy_matches_scalar) {
    std::vector<double> x{1, 2, 3, 4};
    std::vector<double> y{4, 3, 2, 1};
    std::vector<double> ref = y;
    for (size_t i = 0; i < ref.size(); ++i) {
        ref[i] += 0.5 * x[i];
    }
    axpy(0.5, x, y);
    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(y[i], ref[i], 1e-12);
    }
}

TEST(SimdTest, exp_map) {
    std::vector<double> x{0.0, 1.0, 2.0};
    std::vector<double> out(3);
    exp_map(x, out);
    EXPECT_NEAR(out[0], 1.0, 1e-12);
    EXPECT_NEAR(out[1], std::exp(1.0), 1e-9);
}
