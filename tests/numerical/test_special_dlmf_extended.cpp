// MathScript: Extended DLMF special-function reference tests
// Cross-checks zeta, eta, polylog, clausen, and debye against independent
// series/quadrature computations computed inline in this file.

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ms/special/special.hpp"

using namespace ms;

namespace {

double clausen_series_ref(double theta, int terms) {
    double sum = 0.0;
    for (int k = 1; k <= terms; ++k) {
        sum += std::sin(static_cast<double>(k) * theta) /
               (static_cast<double>(k) * static_cast<double>(k));
    }
    return sum;
}

double polylog_series_ref(int n, double z, int terms) {
    double sum = 0.0;
    double zk = z;
    for (int k = 1; k <= terms; ++k) {
        sum += zk / std::pow(static_cast<double>(k), static_cast<double>(n));
        zk *= z;
    }
    return sum;
}

double debye_integrand_ref(int n, double t) {
    if (t <= 1e-15) {
        return (n == 1) ? 1.0 : 0.0;
    }
    return std::pow(t, static_cast<double>(n)) / (std::exp(t) - 1.0);
}

double debye_simpson_ref(int n, double x, int steps) {
    const double h = x / static_cast<double>(steps);
    double sum = 0.0;
    for (int i = 0; i <= steps; ++i) {
        const double t = h * static_cast<double>(i);
        const double weight = (i == 0 || i == steps) ? 1.0 : ((i % 2 == 0) ? 2.0 : 4.0);
        sum += weight * debye_integrand_ref(n, t);
    }
    return static_cast<double>(n) / std::pow(x, static_cast<double>(n)) * sum * h / 3.0;
}

} // namespace

// ---------------------------------------------------------------------------
// Riemann zeta ζ(s)
// ---------------------------------------------------------------------------

TEST(SpecialDlmfExtended, Zeta_KnownClosedForms) {
    const double pi2 = M_PI * M_PI;
    EXPECT_NEAR(zeta(2.0), pi2 / 6.0, 1e-12);
    EXPECT_NEAR(zeta(4.0), pi2 * pi2 / 90.0, 1e-12);
}

TEST(SpecialDlmfExtended, Zeta_AperyConstant) {
    // ζ(3) ≈ 1.2020569031595942 (Apéry's constant)
    EXPECT_NEAR(zeta(3.0), 1.2020569031595942, 1e-9);
}

TEST(SpecialDlmfExtended, Zeta_LargerS) {
    EXPECT_NEAR(zeta(5.0), 1.0369277551433699, 1e-9);
    EXPECT_NEAR(zeta(6.0), 1.0173430619845494, 1e-9);
}

TEST(SpecialDlmfExtended, Zeta_PoleAtOne) {
    EXPECT_TRUE(std::isinf(zeta(1.0)));
}

TEST(SpecialDlmfExtended, Zeta_InvalidDomain) {
    EXPECT_TRUE(std::isnan(zeta(-0.5)));
}

// ---------------------------------------------------------------------------
// Dirichlet eta η(s)
// ---------------------------------------------------------------------------

TEST(SpecialDlmfExtended, EtaDirichlet_KnownValue) {
    const double pi2 = M_PI * M_PI;
    EXPECT_NEAR(eta_dirichlet(2.0), pi2 / 12.0, 1e-12);
}

TEST(SpecialDlmfExtended, EtaDirichlet_ZetaRelation) {
    for (double s : {2.0, 3.0, 4.0, 5.0}) {
        const double expected = (1.0 - std::pow(2.0, 1.0 - s)) * zeta(s);
        EXPECT_NEAR(eta_dirichlet(s), expected, 1e-12) << "eta-zeta relation failed at s=" << s;
    }
}

TEST(SpecialDlmfExtended, EtaDirichlet_AlternatingSeries) {
    double ref = 0.0;
    for (int n = 1; n <= 200000; ++n) {
        ref += (n % 2 == 1 ? 1.0 : -1.0) / std::pow(static_cast<double>(n), 3.0);
    }
    EXPECT_NEAR(eta_dirichlet(3.0), ref, 1e-9);
}

TEST(SpecialDlmfExtended, EtaDirichlet_InvalidDomain) {
    EXPECT_TRUE(std::isnan(eta_dirichlet(-0.1)));
}

// ---------------------------------------------------------------------------
// Polylogarithm Li_n(z)
// ---------------------------------------------------------------------------

TEST(SpecialDlmfExtended, Polylog_N1_LogForm) {
    EXPECT_NEAR(polylog(1, 0.5), -std::log(0.5), 1e-12);
    EXPECT_NEAR(polylog(1, -0.3), -std::log(1.3), 1e-12);
}

TEST(SpecialDlmfExtended, Polylog_N2_ReferenceSeries) {
    const double ref = polylog_series_ref(2, 0.5, 300000);
    EXPECT_NEAR(polylog(2, 0.5), ref, 1e-9);
    EXPECT_NEAR(polylog(2, 0.9), polylog_series_ref(2, 0.9, 500000), 1e-8);
}

TEST(SpecialDlmfExtended, Polylog_N3_ReferenceSeries) {
    EXPECT_NEAR(polylog(3, 0.4), polylog_series_ref(3, 0.4, 200000), 1e-9);
}

TEST(SpecialDlmfExtended, Polylog_BoundaryRejected) {
    EXPECT_TRUE(std::isnan(polylog(2, 1.0)));
    EXPECT_TRUE(std::isnan(polylog(2, 1.5)));
}

TEST(SpecialDlmfExtended, Polylog_InvalidOrder) {
    EXPECT_TRUE(std::isnan(polylog(0, 0.5)));
}

// ---------------------------------------------------------------------------
// Clausen function Cl_2(θ)
// ---------------------------------------------------------------------------

TEST(SpecialDlmfExtended, Clausen_PiOver2) {
    const double ref = clausen_series_ref(M_PI / 2.0, 500000);
    EXPECT_NEAR(clausen(M_PI / 2.0), ref, 1e-12);
}

TEST(SpecialDlmfExtended, Clausen_MultipleAngles) {
    for (double theta : {0.3, 1.0, M_PI / 3.0}) {
        const double ref = clausen_series_ref(theta, 500000);
        EXPECT_NEAR(clausen(theta), ref, 1e-12) << "clausen mismatch at theta=" << theta;
    }
}

TEST(SpecialDlmfExtended, Clausen_Periodicity) {
    const double x = 0.7;
    EXPECT_NEAR(clausen(2.0 * M_PI + x), clausen(x), 1e-9);
    EXPECT_NEAR(clausen(-x), -clausen(x), 1e-9);
}

TEST(SpecialDlmfExtended, Clausen_AtZero) {
    EXPECT_NEAR(clausen(0.0), 0.0, 1e-15);
}

// ---------------------------------------------------------------------------
// Debye function D_n(x)
// ---------------------------------------------------------------------------

TEST(SpecialDlmfExtended, Debye_SmallXLimit) {
    EXPECT_NEAR(debye(1, 1e-6), 1.0, 1e-6);
    EXPECT_NEAR(debye(2, 1e-6), 1.0, 1e-6);
    EXPECT_NEAR(debye(3, 1e-6), 1.0, 1e-6);
}

TEST(SpecialDlmfExtended, Debye_CrossCheckSimpson) {
    for (int n : {1, 2, 3}) {
        for (double x : {0.5, 1.0, 2.0, 5.0}) {
            const double ref = debye_simpson_ref(n, x, 1024);
            EXPECT_NEAR(debye(n, x), ref, 1e-9) << "debye(" << n << ", " << x << ") mismatch";
        }
    }
}

TEST(SpecialDlmfExtended, Debye_MonotoneDecreasing) {
    EXPECT_GT(debye(2, 0.5), debye(2, 2.0));
    EXPECT_GT(debye(3, 1.0), debye(3, 4.0));
}

TEST(SpecialDlmfExtended, Debye_InvalidDomain) {
    EXPECT_TRUE(std::isnan(debye(0, 1.0)));
    EXPECT_TRUE(std::isnan(debye(2, 0.0)));
    EXPECT_TRUE(std::isnan(debye(2, -1.0)));
}
