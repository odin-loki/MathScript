#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

#include <ms/special/special.hpp>

struct SpecialRef {
    double x;
    double expected;
    double tol;
};

TEST(NumericalSpecial, Erf_DLMF_Reference) {
    EXPECT_NEAR(ms::erf(0.0), 0.0, 1e-15);
    EXPECT_NEAR(ms::erf(1.0), 0.8427007929, 1e-9);
    EXPECT_NEAR(ms::erf(2.0), 0.9953222650, 1e-9);
    EXPECT_NEAR(ms::erf(std::numeric_limits<double>::infinity()), 1.0, 1e-15);
}

TEST(NumericalSpecial, Erfc_DLMF_Reference) {
    EXPECT_NEAR(ms::erfc(0.0), 1.0, 1e-15);
    EXPECT_NEAR(ms::erfc(1.0), 0.1572992070, 1e-9);
    EXPECT_NEAR(ms::erfc(2.0), 0.0046777350, 1e-9);
    EXPECT_NEAR(ms::erfc(std::numeric_limits<double>::infinity()), 0.0, 1e-15);
}

TEST(NumericalSpecial, Gamma_DLMF_Reference) {
    EXPECT_NEAR(ms::gamma_func(1.0), 1.0, 1e-12);
    EXPECT_NEAR(ms::gamma_func(0.5), std::sqrt(std::numbers::pi), 1e-12);
    EXPECT_NEAR(ms::gamma_func(5.0), 24.0, 1e-12);
}

TEST(NumericalSpecial, LogGamma_DLMF_Reference) {
    EXPECT_NEAR(ms::log_gamma(1.0), 0.0, 1e-12);
    EXPECT_NEAR(ms::log_gamma(10.0), 12.8018274801, 1e-9);
    EXPECT_NEAR(ms::log_gamma(0.5), 0.5 * std::log(std::numbers::pi), 1e-12);
}

TEST(NumericalSpecial, Tgamma_MatchesGamma) {
    static const std::vector<SpecialRef> REF = {
        {1.0, 1.0, 1e-12},
        {0.5, std::sqrt(std::numbers::pi), 1e-12},
        {5.0, 24.0, 1e-12},
        {2.5, 1.3293403882, 1e-9},
    };
    for (const auto& ref : REF) {
        EXPECT_NEAR(ms::gamma_func(ref.x), ref.expected, ref.tol)
            << "gamma(" << ref.x << ")";
    }
}

TEST(NumericalSpecial, Digamma_DLMF_Reference) {
    EXPECT_NEAR(ms::digamma(1.0), -0.5772156649, 1e-9);
    EXPECT_NEAR(ms::digamma(2.0), 0.4227843351, 1e-9);
    EXPECT_NEAR(ms::digamma(0.5), -1.9635100260, 1e-9);
}
