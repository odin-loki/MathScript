#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

#include "ms/special/special.hpp"

using namespace ms;

namespace {

double normalized_gaussian(double x, double sigma) {
    return std::exp(-x * x / (2.0 * sigma * sigma)) / (sigma * std::sqrt(2.0 * std::numbers::pi));
}

double normalized_lorentzian(double x, double gamma) {
    return gamma / (std::numbers::pi * (x * x + gamma * gamma));
}

// Simple Simpson's-rule integral of voigt(x, sigma, gamma) over [-limit, limit] with n
// (even) subintervals; used as an independent end-to-end normalization check.
double integrate_voigt(double sigma, double gamma, double limit, int n) {
    if (n % 2 != 0) {
        ++n;
    }
    const double h = (2.0 * limit) / static_cast<double>(n);
    double sum = voigt(-limit, sigma, gamma) + voigt(limit, sigma, gamma);
    for (int i = 1; i < n; ++i) {
        const double x = -limit + i * h;
        sum += (i % 2 == 0 ? 2.0 : 4.0) * voigt(x, sigma, gamma);
    }
    return sum * h / 3.0;
}

double thompson_cox_hastings_eta(double sigma, double gamma) {
    const double f_g = 2.0 * sigma * std::sqrt(2.0 * std::log(2.0));
    const double f_l = 2.0 * gamma;
    const double f = std::pow(std::pow(f_g, 5.0) + 2.69269 * std::pow(f_g, 4.0) * f_l +
                                   2.42843 * std::pow(f_g, 3.0) * f_l * f_l +
                                   4.47163 * f_g * f_g * std::pow(f_l, 3.0) +
                                   0.07842 * f_g * std::pow(f_l, 4.0) + std::pow(f_l, 5.0),
                               0.2);
    const double r = f_l / f;
    return 1.36603 * r - 0.47719 * r * r + 0.11116 * r * r * r;
}

} // namespace

TEST(SpecialVoigtTest, gamma_zero_reduces_to_gaussian) {
    // The Humlicek w4 rational approximation has ~1e-4 relative accuracy in its region II
    // (moderate |x|+y), so an absolute tolerance scaled to the local Gaussian value is used.
    for (double sigma : {0.5, 1.0, 2.0}) {
        for (double x : {0.0, 0.3, 1.0, -1.5}) {
            const double expected = normalized_gaussian(x, sigma);
            EXPECT_NEAR(voigt(x, sigma, 0.0), expected, 1e-4 * expected + 1e-8);
        }
    }
}

TEST(SpecialVoigtTest, tiny_sigma_reduces_to_lorentzian) {
    const double sigma = 1e-6;
    for (double gamma : {0.5, 1.0, 2.0}) {
        for (double x : {0.0, 0.5, 1.0, -2.0}) {
            EXPECT_NEAR(voigt(x, sigma, gamma), normalized_lorentzian(x, gamma), 1e-3);
        }
    }
}

TEST(SpecialVoigtTest, normalization_integral_pure_gaussian) {
    const double integral = integrate_voigt(1.0, 0.0, 50.0, 20000);
    EXPECT_NEAR(integral, 1.0, 0.02);
}

TEST(SpecialVoigtTest, normalization_integral_mixed_cases) {
    struct Case {
        double sigma;
        double gamma;
    };
    const Case cases[] = {{1.0, 1.0}, {0.5, 2.0}, {2.0, 0.5}, {1.0, 0.1}};
    for (const auto& c : cases) {
        const double limit = 50.0 * std::max(c.sigma, c.gamma);
        const double integral = integrate_voigt(c.sigma, c.gamma, limit, 40000);
        EXPECT_NEAR(integral, 1.0, 0.03);
    }
}

TEST(SpecialVoigtTest, normalization_integral_pure_lorentzian_like) {
    const double integral = integrate_voigt(1e-6, 1.0, 500.0, 200000);
    EXPECT_NEAR(integral, 1.0, 0.05);
}

TEST(SpecialVoigtTest, symmetry_even_function) {
    for (double x : {0.1, 0.5, 1.0, 2.5, 5.0}) {
        EXPECT_NEAR(voigt(x, 1.0, 0.7), voigt(-x, 1.0, 0.7), 1e-12);
        EXPECT_NEAR(voigt(x, 0.3, 1.5), voigt(-x, 0.3, 1.5), 1e-12);
    }
}

TEST(SpecialVoigtTest, peak_at_center) {
    const double sigma = 0.8;
    const double gamma = 0.6;
    const double peak = voigt(0.0, sigma, gamma);
    for (double x = -5.0; x <= 5.0; x += 0.05) {
        if (x == 0.0) {
            continue;
        }
        EXPECT_LE(voigt(x, sigma, gamma), peak);
    }
}

TEST(SpecialVoigtTest, decays_at_large_x) {
    EXPECT_NEAR(voigt(1000.0, 1.0, 1.0), 0.0, 1e-6);
    EXPECT_NEAR(voigt(-1000.0, 0.5, 2.0), 0.0, 1e-6);
}

TEST(SpecialVoigtTest, gamma_negative_clamped_to_zero) {
    EXPECT_NEAR(voigt(0.4, 1.0, -1.0), voigt(0.4, 1.0, 0.0), 1e-10);
}

TEST(SpecialVoigtTest, nonpositive_sigma_falls_back_to_lorentzian) {
    EXPECT_NEAR(voigt(0.5, 0.0, 1.0), normalized_lorentzian(0.5, 1.0), 1e-10);
    EXPECT_NEAR(voigt(0.5, -1.0, 1.0), normalized_lorentzian(0.5, 1.0), 1e-10);
}

TEST(SpecialVoigtTest, degenerate_sigma_and_gamma_zero_is_finite_or_spike) {
    EXPECT_TRUE(std::isinf(voigt(0.0, 0.0, 0.0)) || voigt(0.0, 0.0, 0.0) >= 0.0);
    EXPECT_NEAR(voigt(1.0, 0.0, 0.0), 0.0, 1e-12);
}

TEST(SpecialVoigtTest, positive_everywhere) {
    for (double x : {-3.0, -1.0, 0.0, 1.0, 3.0}) {
        EXPECT_GT(voigt(x, 1.0, 0.5), 0.0);
    }
}

TEST(SpecialVoigtTest, pseudo_voigt_symmetry) {
    for (double x : {0.2, 0.9, 2.0}) {
        EXPECT_NEAR(pseudo_voigt(x, 1.0, 0.5, 0.4), pseudo_voigt(-x, 1.0, 0.5, 0.4), 1e-12);
        EXPECT_NEAR(pseudo_voigt_auto(x, 1.0, 0.5), pseudo_voigt_auto(-x, 1.0, 0.5), 1e-12);
    }
}

TEST(SpecialVoigtTest, pseudo_voigt_eta_zero_is_shared_fwhm_gaussian) {
    const double sigma = 1.0;
    const double gamma = 0.5;
    const double f_g = 2.0 * sigma * std::sqrt(2.0 * std::log(2.0));
    const double f_l = 2.0 * gamma;
    const double f = std::pow(std::pow(f_g, 5.0) + 2.69269 * std::pow(f_g, 4.0) * f_l +
                                   2.42843 * std::pow(f_g, 3.0) * f_l * f_l +
                                   4.47163 * f_g * f_g * std::pow(f_l, 3.0) +
                                   0.07842 * f_g * std::pow(f_l, 4.0) + std::pow(f_l, 5.0),
                               0.2);
    const double sigma_pv = f / (2.0 * std::sqrt(2.0 * std::log(2.0)));
    for (double x : {0.0, 0.5, 1.5}) {
        EXPECT_NEAR(pseudo_voigt(x, sigma, gamma, 0.0), normalized_gaussian(x, sigma_pv), 1e-9);
    }
}

TEST(SpecialVoigtTest, pseudo_voigt_eta_one_is_shared_fwhm_lorentzian) {
    const double sigma = 1.0;
    const double gamma = 0.5;
    const double f_g = 2.0 * sigma * std::sqrt(2.0 * std::log(2.0));
    const double f_l = 2.0 * gamma;
    const double f = std::pow(std::pow(f_g, 5.0) + 2.69269 * std::pow(f_g, 4.0) * f_l +
                                   2.42843 * std::pow(f_g, 3.0) * f_l * f_l +
                                   4.47163 * f_g * f_g * std::pow(f_l, 3.0) +
                                   0.07842 * f_g * std::pow(f_l, 4.0) + std::pow(f_l, 5.0),
                               0.2);
    const double gamma_pv = f / 2.0;
    for (double x : {0.0, 0.5, 1.5}) {
        EXPECT_NEAR(pseudo_voigt(x, sigma, gamma, 1.0), normalized_lorentzian(x, gamma_pv), 1e-9);
    }
}

TEST(SpecialVoigtTest, pseudo_voigt_auto_matches_manual_eta) {
    const double sigma = 0.7;
    const double gamma = 1.3;
    const double eta = thompson_cox_hastings_eta(sigma, gamma);
    for (double x : {-2.0, 0.0, 0.3, 4.0}) {
        EXPECT_NEAR(pseudo_voigt_auto(x, sigma, gamma), pseudo_voigt(x, sigma, gamma, eta), 1e-12);
    }
}

TEST(SpecialVoigtTest, pseudo_voigt_eta_clamped) {
    EXPECT_NEAR(pseudo_voigt(0.3, 1.0, 0.5, -0.5), pseudo_voigt(0.3, 1.0, 0.5, 0.0), 1e-12);
    EXPECT_NEAR(pseudo_voigt(0.3, 1.0, 0.5, 1.5), pseudo_voigt(0.3, 1.0, 0.5, 1.0), 1e-12);
}

TEST(SpecialVoigtTest, pseudo_voigt_peak_at_center) {
    const double peak = pseudo_voigt_auto(0.0, 0.6, 0.9);
    for (double x = -4.0; x <= 4.0; x += 0.1) {
        if (x == 0.0) {
            continue;
        }
        EXPECT_LE(pseudo_voigt_auto(x, 0.6, 0.9), peak);
    }
}

TEST(SpecialVoigtTest, pseudo_voigt_decays_at_large_x) {
    EXPECT_NEAR(pseudo_voigt_auto(1000.0, 1.0, 1.0), 0.0, 1e-6);
}

TEST(SpecialVoigtTest, pseudo_voigt_eta_in_unit_range_for_auto) {
    for (double sigma : {0.1, 1.0, 5.0}) {
        for (double gamma : {0.0, 0.5, 3.0}) {
            const double eta = thompson_cox_hastings_eta(sigma, gamma);
            EXPECT_GE(eta, -1e-9);
            EXPECT_LE(eta, 1.0 + 1e-9);
        }
    }
}

TEST(SpecialVoigtTest, voigt_at_zero_matches_expected_scale) {
    // At x=0, voigt reduces to Re[w(i*a)]/(sigma*sqrt(2*pi)) with a = gamma/(sigma*sqrt(2)).
    // For gamma=0 this must equal the Gaussian peak 1/(sigma*sqrt(2*pi)).
    const double sigma = 1.2;
    EXPECT_NEAR(voigt(0.0, sigma, 0.0), 1.0 / (sigma * std::sqrt(2.0 * std::numbers::pi)), 1e-9);
}

TEST(SpecialVoigtTest, voigt_reasonable_relative_to_gaussian_and_lorentzian_bounds) {
    // The Voigt profile at x=0 should lie between the (lower) pure-Lorentzian-like spread and
    // the (higher) pure-Gaussian peak scale for moderate mixed parameters -- a coarse sanity
    // bound that catches gross implementation errors without relying on a precise reference.
    const double sigma = 1.0;
    const double gamma = 1.0;
    const double v0 = voigt(0.0, sigma, gamma);
    EXPECT_GT(v0, 0.0);
    EXPECT_LT(v0, 1.0 / (sigma * std::sqrt(2.0 * std::numbers::pi)));
}
