#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>

#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"
#include "ms/special/special.hpp"

using namespace ms;

namespace {

uint32_t lcg_next(uint32_t& state) {
    state = state * 1664525U + 1013904223U;
    return state;
}

double random_unit(uint32_t& state) {
    const uint32_t bits = lcg_next(state);
    return static_cast<double>(bits) / static_cast<double>(std::numeric_limits<uint32_t>::max()) * 2.0 - 1.0;
}

double random_positive(uint32_t& state, double max_abs) {
    return (random_unit(state) * 0.5 + 0.5) * max_abs;
}

bool is_safe(double x) {
    return !std::isnan(x);
}

} // namespace

TEST(FuzzSpecialFnsTest, random_inputs_do_not_crash) {
    uint32_t state = 0xC0FFEEU;
    for (int i = 0; i < 512; ++i) {
        const double x = random_unit(state) * 4.0;
        const double s = random_positive(state, 3.0) + 0.5;
        const double q = random_positive(state, 0.4);
        const double z = random_positive(state, 0.3);

        EXPECT_TRUE(is_safe(ms::erf(x)));
        EXPECT_TRUE(is_safe(ms::gamma_func(std::abs(x) + 0.2)));
        EXPECT_TRUE(is_safe(ms::bessel_j0(x)));
        EXPECT_TRUE(is_safe(ms::zeta(s)));
        EXPECT_TRUE(is_safe(ms::polylog(2, random_unit(state) * 0.8)));
        EXPECT_TRUE(is_safe(ms::theta3(z, q)));
        EXPECT_TRUE(is_safe(ms::mathieu_ce(1, 0.1, x)));
        EXPECT_TRUE(is_safe(ms::heun_c(0.1, 0.2, 0.3, 0.4, 0.5, random_positive(state, 0.15))));
        EXPECT_TRUE(is_safe(ms::painleve1(random_positive(state, 0.8), 0.0, 0.0)));
    }
}

TEST(FuzzMatrixOpsTest, random_matmul_shapes) {
    uint32_t state = 0xFACEB00CU;
    for (int i = 0; i < 64; ++i) {
        const std::size_t m = 1 + static_cast<std::size_t>(lcg_next(state) % 12);
        const std::size_t k = 1 + static_cast<std::size_t>(lcg_next(state) % 12);
        const std::size_t n = 1 + static_cast<std::size_t>(lcg_next(state) % 12);
        ColMatrix<double> A(m, k);
        ColMatrix<double> B(k, n);
        for (std::size_t r = 0; r < m; ++r) {
            for (std::size_t c = 0; c < k; ++c) {
                A(r, c) = random_unit(state);
            }
        }
        for (std::size_t r = 0; r < k; ++r) {
            for (std::size_t c = 0; c < n; ++c) {
                B(r, c) = random_unit(state);
            }
        }

        auto result = matmul(A, B);
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->rows(), m);
        EXPECT_EQ(result->cols(), n);
        for (std::size_t r = 0; r < m; ++r) {
            for (std::size_t c = 0; c < n; ++c) {
                EXPECT_TRUE(is_safe((*result)(r, c)));
            }
        }
    }
}
