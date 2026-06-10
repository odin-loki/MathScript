#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <numbers>

#include <ms/special/special.hpp>

using namespace ms;

// ============================================================
// Group 1: Chebyshev polynomials
// chebyshev_t(int n, double x)   — T_n(x), first kind
// chebyshev_u(int n, double x)   — U_n(x), second kind
// chebyshev_tn(int n, int k, double x) — variant with extra parameter
// chebyshev_un(int n, int k, double x) — variant with extra parameter
// chebyshev_v(int n, double x)   — third kind
// chebyshev_w(int n, double x)   — fourth kind
// ============================================================

TEST(SpecialGroup1, ChebyshevT_KnownValues) {
    // T_0(x) = 1
    EXPECT_NEAR(chebyshev_t(0, 0.5),  1.0, 1e-12);
    EXPECT_NEAR(chebyshev_t(0, -0.7), 1.0, 1e-12);
    // T_1(x) = x
    EXPECT_NEAR(chebyshev_t(1,  0.5),  0.5,  1e-12);
    EXPECT_NEAR(chebyshev_t(1, -0.3), -0.3,  1e-12);
    // T_2(x) = 2x^2 - 1
    EXPECT_NEAR(chebyshev_t(2, 0.5), 2.0*0.25 - 1.0, 1e-12);
    EXPECT_NEAR(chebyshev_t(2, 0.8), 2.0*0.64 - 1.0, 1e-12);
    // T_3(x) = 4x^3 - 3x
    EXPECT_NEAR(chebyshev_t(3, 0.5), 4.0*0.125 - 1.5, 1e-12);
}

TEST(SpecialGroup1, ChebyshevT_BoundaryPoints) {
    // T_n(1) = 1 for all n
    for (int n = 0; n <= 5; ++n) {
        EXPECT_NEAR(chebyshev_t(n, 1.0), 1.0, 1e-12) << "T_" << n << "(1)";
    }
    // T_n(-1) = (-1)^n
    for (int n = 0; n <= 5; ++n) {
        double expected = (n % 2 == 0) ? 1.0 : -1.0;
        EXPECT_NEAR(chebyshev_t(n, -1.0), expected, 1e-12) << "T_" << n << "(-1)";
    }
}

TEST(SpecialGroup1, ChebyshevT_Symmetry) {
    // T_n is even for even n, odd for odd n
    for (double x : {0.3, 0.6, 0.9}) {
        EXPECT_NEAR(chebyshev_t(2, x), chebyshev_t(2, -x), 1e-12);
        EXPECT_NEAR(chebyshev_t(3, x), -chebyshev_t(3, -x), 1e-12);
        EXPECT_NEAR(chebyshev_t(4, x), chebyshev_t(4, -x), 1e-12);
    }
}

TEST(SpecialGroup1, ChebyshevU_KnownValues) {
    // U_0(x) = 1
    EXPECT_NEAR(chebyshev_u(0, 0.5), 1.0, 1e-12);
    // U_1(x) = 2x
    EXPECT_NEAR(chebyshev_u(1, 0.5), 1.0, 1e-12);
    EXPECT_NEAR(chebyshev_u(1, 0.3), 0.6, 1e-12);
    // U_2(x) = 4x^2 - 1
    EXPECT_NEAR(chebyshev_u(2, 0.5), 4.0*0.25 - 1.0, 1e-12);
    EXPECT_NEAR(chebyshev_u(2, 0.8), 4.0*0.64 - 1.0, 1e-12);
}

TEST(SpecialGroup1, ChebyshevU_BoundaryPoints) {
    // U_n(1) = n + 1
    for (int n = 0; n <= 4; ++n) {
        EXPECT_NEAR(chebyshev_u(n, 1.0), static_cast<double>(n + 1), 1e-10)
            << "U_" << n << "(1)";
    }
}

TEST(SpecialGroup1, ChebyshevTn_Smoke) {
    // chebyshev_tn(int n, int k, double x) — verify finite output
    EXPECT_TRUE(std::isfinite(chebyshev_tn(3, 0, 0.5)));
    EXPECT_TRUE(std::isfinite(chebyshev_tn(3, 1, 0.5)));
    EXPECT_TRUE(std::isfinite(chebyshev_tn(5, 0, -0.3)));
    EXPECT_TRUE(std::isfinite(chebyshev_tn(4, 2,  0.7)));
}

TEST(SpecialGroup1, ChebyshevUn_Smoke) {
    // chebyshev_un(int n, int k, double x) — verify finite output
    EXPECT_TRUE(std::isfinite(chebyshev_un(3, 0, 0.5)));
    EXPECT_TRUE(std::isfinite(chebyshev_un(3, 1, 0.5)));
    EXPECT_TRUE(std::isfinite(chebyshev_un(4, 0, -0.3)));
    EXPECT_TRUE(std::isfinite(chebyshev_un(5, 2,  0.7)));
}

TEST(SpecialGroup1, ChebyshevVW_KnownValues) {
    // V_0(x) and W_0(x) have implementation-dependent normalization
    // Just check finite values
    EXPECT_TRUE(std::isfinite(chebyshev_v(0, 0.5)));
    EXPECT_TRUE(std::isfinite(chebyshev_w(0, 0.5)));
    // Verify finite for higher orders
    EXPECT_TRUE(std::isfinite(chebyshev_v(1, 0.5)));
    EXPECT_TRUE(std::isfinite(chebyshev_v(3, 0.5)));
    EXPECT_TRUE(std::isfinite(chebyshev_w(1, 0.5)));
    EXPECT_TRUE(std::isfinite(chebyshev_w(3, 0.5)));
}

// ============================================================
// Group 2: Legendre polynomials
// legendre_p(int n, double x)          — P_n(x)
// legendre_q(int n, double x)          — Q_n(x)
// legendre_pn(int n, int m, double x)  — associated P_n^m(x)
// ============================================================

TEST(SpecialGroup1, LegendreP_KnownValues) {
    // P_0(x) = 1
    EXPECT_NEAR(legendre_p(0, 0.5), 1.0, 1e-12);
    // P_1(x) = x
    EXPECT_NEAR(legendre_p(1,  0.5),  0.5, 1e-12);
    EXPECT_NEAR(legendre_p(1, -0.3), -0.3, 1e-12);
    // P_2(x) = (3x^2 - 1) / 2
    EXPECT_NEAR(legendre_p(2, 0.5), (3.0*0.25 - 1.0) / 2.0, 1e-12);
    EXPECT_NEAR(legendre_p(2, 0.8), (3.0*0.64 - 1.0) / 2.0, 1e-12);
    // P_3(x) = (5x^3 - 3x) / 2
    EXPECT_NEAR(legendre_p(3, 0.5), (5.0*0.125 - 1.5) / 2.0, 1e-12);
}

TEST(SpecialGroup1, LegendreP_BoundaryPoints) {
    // P_n(1) = 1 for all n
    for (int n = 0; n <= 5; ++n) {
        EXPECT_NEAR(legendre_p(n, 1.0), 1.0, 1e-11) << "P_" << n << "(1)";
    }
    // P_n(-1) = (-1)^n
    for (int n = 0; n <= 5; ++n) {
        double expected = (n % 2 == 0) ? 1.0 : -1.0;
        EXPECT_NEAR(legendre_p(n, -1.0), expected, 1e-11) << "P_" << n << "(-1)";
    }
}

TEST(SpecialGroup1, LegendreP_Symmetry) {
    // P_n(-x) = (-1)^n P_n(x)
    for (double x : {0.3, 0.6, 0.9}) {
        EXPECT_NEAR(legendre_p(2, -x),  legendre_p(2, x), 1e-12);  // even
        EXPECT_NEAR(legendre_p(3, -x), -legendre_p(3, x), 1e-12);  // odd
        EXPECT_NEAR(legendre_p(4, -x),  legendre_p(4, x), 1e-12);  // even
    }
}

TEST(SpecialGroup1, LegendreP_Recurrence) {
    // (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
    for (double x : {0.3, 0.6, -0.5}) {
        for (int n = 1; n <= 4; ++n) {
            double lhs = static_cast<double>(n + 1) * legendre_p(n + 1, x);
            double rhs = static_cast<double>(2*n + 1) * x * legendre_p(n, x)
                       - static_cast<double>(n) * legendre_p(n - 1, x);
            EXPECT_NEAR(lhs, rhs, 1e-10) << "n=" << n << " x=" << x;
        }
    }
}

TEST(SpecialGroup1, LegendreQ_KnownValues) {
    // Q_0(x) = (1/2) * ln((1+x)/(1-x))  for |x| < 1 — may vary by convention
    // Smoke test: check finite and correct sign pattern
    for (double x : {0.3, 0.5, 0.7}) {
        double q0 = legendre_q(0, x);
        EXPECT_TRUE(std::isfinite(q0)) << "Q_0(" << x << ") should be finite";
        EXPECT_GT(q0, 0.0) << "Q_0(" << x << ") should be positive for 0 < x < 1";
    }
    // Q_1(x) should be finite for x in (-1, 1)
    {
        double x = 0.5;
        double q1 = legendre_q(1, x);
        EXPECT_TRUE(std::isfinite(q1));
    }
}

TEST(SpecialGroup1, LegendreQ_Finite) {
    for (int n = 0; n <= 4; ++n) {
        EXPECT_TRUE(std::isfinite(legendre_q(n, 0.5))) << "Q_" << n << "(0.5)";
    }
}

TEST(SpecialGroup1, LegendrePn_Associated) {
    // P_0^0(x) = 1
    EXPECT_NEAR(legendre_pn(0, 0, 0.5), 1.0, 1e-12);
    // P_1^0(x) = x
    EXPECT_NEAR(legendre_pn(1, 0, 0.5), 0.5, 1e-12);
    // P_2^0(x) = (3x^2 - 1) / 2
    EXPECT_NEAR(legendre_pn(2, 0, 0.5), (3.0*0.25 - 1.0) / 2.0, 1e-12);
    // Higher-order associated terms are finite
    EXPECT_TRUE(std::isfinite(legendre_pn(1, 1, 0.5)));
    EXPECT_TRUE(std::isfinite(legendre_pn(2, 1, 0.5)));
    EXPECT_TRUE(std::isfinite(legendre_pn(2, 2, 0.5)));
}

TEST(SpecialGroup1, LegendrePn_Smoke) {
    for (int n = 0; n <= 4; ++n) {
        for (int m = 0; m <= n; ++m) {
            EXPECT_TRUE(std::isfinite(legendre_pn(n, m, 0.5)))
                << "P_" << n << "^" << m << "(0.5)";
        }
    }
}

// ============================================================
// Group 3: Laguerre polynomials
// laguerre_l(int n, double x)               — L_n(x)
// laguerre_la(int n, double a, double x)    — generalized L_n^a(x)
// laguerre_ln(int n, int k, double x)       — variant
// ============================================================

TEST(SpecialGroup1, LaguerreL_KnownValues) {
    // L_0(x) = 1
    EXPECT_NEAR(laguerre_l(0, 0.5), 1.0, 1e-12);
    EXPECT_NEAR(laguerre_l(0, 2.0), 1.0, 1e-12);
    // L_1(x) = 1 - x
    EXPECT_NEAR(laguerre_l(1, 0.5),  0.5, 1e-12);
    EXPECT_NEAR(laguerre_l(1, 2.0), -1.0, 1e-12);
    // L_2(x) = (x^2 - 4x + 2) / 2
    EXPECT_NEAR(laguerre_l(2, 0.0),  1.0, 1e-12);   // (0-0+2)/2 = 1
    EXPECT_NEAR(laguerre_l(2, 1.0), -0.5, 1e-12);   // (1-4+2)/2 = -0.5
    EXPECT_NEAR(laguerre_l(2, 2.0), -1.0, 1e-12);   // (4-8+2)/2 = -1
}

TEST(SpecialGroup1, LaguerreL_AtZero) {
    // L_n(0) = 1 for all n
    for (int n = 0; n <= 5; ++n) {
        EXPECT_NEAR(laguerre_l(n, 0.0), 1.0, 1e-12) << "L_" << n << "(0)";
    }
}

TEST(SpecialGroup1, LaguerreL_Recurrence) {
    // (n+1) L_{n+1}(x) = (2n+1-x) L_n(x) - n L_{n-1}(x)
    for (double x : {0.5, 1.0, 2.0}) {
        for (int n = 1; n <= 3; ++n) {
            double lhs = static_cast<double>(n + 1) * laguerre_l(n + 1, x);
            double rhs = (2.0*n + 1.0 - x) * laguerre_l(n, x)
                       - static_cast<double>(n) * laguerre_l(n - 1, x);
            EXPECT_NEAR(lhs, rhs, 1e-10) << "n=" << n << " x=" << x;
        }
    }
}

TEST(SpecialGroup1, LaguerreLa_KnownValues) {
    // L_0^a(x) = 1 for all a, x
    EXPECT_NEAR(laguerre_la(0, 1.0, 0.5), 1.0, 1e-12);
    EXPECT_NEAR(laguerre_la(0, 2.0, 0.5), 1.0, 1e-12);
    // L_1^a(x) = 1 + a - x
    EXPECT_NEAR(laguerre_la(1, 1.0, 0.5), 1.5,  1e-12);  // 1+1-0.5=1.5
    EXPECT_NEAR(laguerre_la(1, 2.0, 1.0), 2.0,  1e-12);  // 1+2-1=2
    EXPECT_NEAR(laguerre_la(1, 0.0, 0.5), 0.5,  1e-12);  // 1+0-0.5=0.5
}

TEST(SpecialGroup1, LaguerreLa_Recurrence) {
    // (n+1) L_{n+1}^a(x) = (2n+a+1-x) L_n^a(x) - (n+a) L_{n-1}^a(x)
    double a = 1.5, x = 0.7;
    double l0 = laguerre_la(0, a, x);
    double l1 = laguerre_la(1, a, x);
    double l2 = laguerre_la(2, a, x);
    // n=1: 2 * l2 = (2+a+1-x)*l1 - (1+a)*l0
    double expected = (3.0 + a - x) * l1 / 2.0 - (1.0 + a) * l0 / 2.0;
    EXPECT_NEAR(l2, expected, 1e-10);
}

TEST(SpecialGroup1, LaguerreLn_Smoke) {
    // laguerre_ln(int n, int k, double x) — verify finite output
    EXPECT_TRUE(std::isfinite(laguerre_ln(2, 0, 0.5)));
    EXPECT_TRUE(std::isfinite(laguerre_ln(3, 1, 0.5)));
    EXPECT_TRUE(std::isfinite(laguerre_ln(4, 0, 1.0)));
    EXPECT_TRUE(std::isfinite(laguerre_ln(5, 2, 0.3)));
}

// ============================================================
// Group 4: Jacobi elliptic functions
// jacobi_am(double u, double k) — amplitude
// jacobi_ns(double u, double k), jacobi_nc(double u, double k)
// jacobi_nd(double u, double k)
// ============================================================

TEST(SpecialGroup1, JacobiAm_AtK0) {
    // In the circular limit k->0: am(u, 0) = u
    for (double u : {0.3, 0.5, 1.0, 1.5}) {
        EXPECT_NEAR(jacobi_am(u, 0.0), u, 1e-10) << "am(" << u << ", 0)";
    }
}

TEST(SpecialGroup1, JacobiNd_Finite) {
    // nd(u, k) should be finite for reasonable inputs
    for (double u : {0.3, 0.5, 1.0, 1.5}) {
        EXPECT_TRUE(std::isfinite(jacobi_nd(u, 0.5))) << "nd(" << u << ", 0.5) not finite";
        EXPECT_TRUE(std::isfinite(jacobi_nd(u, 0.0))) << "nd(" << u << ", 0) not finite";
    }
}

TEST(SpecialGroup1, JacobiNc_Finite) {
    // nc(u, k) should be finite for reasonable inputs
    for (double u : {0.3, 0.5, 0.8}) {
        EXPECT_TRUE(std::isfinite(jacobi_nc(u, 0.5))) << "nc(" << u << ", 0.5) not finite";
        EXPECT_TRUE(std::isfinite(jacobi_nc(u, 0.0))) << "nc(" << u << ", 0) not finite";
    }
}

TEST(SpecialGroup1, JacobiNs_AtK0) {
    // sn(u, 0) = sin(u), so ns(u, 0) = 1/sin(u)
    for (double u : {0.3, 0.5, 0.8}) {
        EXPECT_NEAR(jacobi_ns(u, 0.0), 1.0 / std::sin(u), 1e-9)
            << "ns(" << u << ", 0)";
    }
}

TEST(SpecialGroup1, JacobiElliptic_Smoke) {
    // Verify finite values for typical arguments
    double u = 0.5, k = 0.5;
    EXPECT_TRUE(std::isfinite(jacobi_am(u, k)));
    EXPECT_TRUE(std::isfinite(jacobi_ns(u, k)));
    EXPECT_TRUE(std::isfinite(jacobi_nc(u, k)));
    EXPECT_TRUE(std::isfinite(jacobi_nd(u, k)));
    EXPECT_TRUE(std::isfinite(jacobi_sn(u, k)));
    EXPECT_TRUE(std::isfinite(jacobi_cn(u, k)));
    EXPECT_TRUE(std::isfinite(jacobi_dn(u, k)));
}

// ============================================================
// Group 5: Theta functions
// theta1(double z, double q), theta2(double z, double q)
// theta3(double z, double q), theta4(double z, double q)
// ============================================================

TEST(SpecialGroup1, Theta1_ZeroAtOrigin) {
    // theta1(0, q) = 0 for all 0 < q < 1
    EXPECT_NEAR(theta1(0.0, 0.1), 0.0, 1e-12);
    EXPECT_NEAR(theta1(0.0, 0.3), 0.0, 1e-12);
    EXPECT_NEAR(theta1(0.0, 0.5), 0.0, 1e-12);
}

TEST(SpecialGroup1, Theta2_Finite) {
    EXPECT_TRUE(std::isfinite(theta2(0.0, 0.3)));
    EXPECT_TRUE(std::isfinite(theta2(0.5, 0.3)));
    EXPECT_TRUE(std::isfinite(theta2(1.0, 0.5)));
    // theta2(0, q) > 0 for 0 < q < 1
    EXPECT_GT(theta2(0.0, 0.3), 0.0);
}

TEST(SpecialGroup1, Theta3_Finite) {
    EXPECT_TRUE(std::isfinite(theta3(0.0, 0.3)));
    EXPECT_TRUE(std::isfinite(theta3(0.5, 0.3)));
    // theta3(0, q) > 0
    EXPECT_GT(theta3(0.0, 0.3), 0.0);
}

TEST(SpecialGroup1, Theta4_Finite) {
    EXPECT_TRUE(std::isfinite(theta4(0.0, 0.3)));
    EXPECT_TRUE(std::isfinite(theta4(0.5, 0.3)));
    EXPECT_TRUE(std::isfinite(theta4(1.0, 0.5)));
}

// ============================================================
// Group 6: Bessel functions (extended)
// bessel_h(int nu, double x) — legacy alias
// bessel_y1(double x)        — Y_1(x)
// struve_hn(int nu, double x), struve_l(int nu, double x)
// struve_h(int nu, double x)
// ============================================================

TEST(SpecialGroup1, BesselH_Smoke) {
    // bessel_h is a legacy alias — verify finite values
    EXPECT_TRUE(std::isfinite(bessel_h(0, 1.0)));
    EXPECT_TRUE(std::isfinite(bessel_h(1, 1.0)));
    EXPECT_TRUE(std::isfinite(bessel_h(2, 2.0)));
    EXPECT_TRUE(std::isfinite(bessel_h(3, 3.0)));
}

TEST(SpecialGroup1, BesselY1_KnownValues) {
    // Y_1(1) is negative (exact value depends on convention; standard is ≈ -0.7812)
    EXPECT_LT(bessel_y1(1.0), 0.0);
    // Finite for x > 0
    EXPECT_TRUE(std::isfinite(bessel_y1(0.5)));
    EXPECT_TRUE(std::isfinite(bessel_y1(2.0)));
    EXPECT_TRUE(std::isfinite(bessel_y1(5.0)));
}

TEST(SpecialGroup1, StruveH_KnownValues) {
    // H_nu(0) = 0 for nu >= 0 (odd function near origin)
    EXPECT_NEAR(struve_h(0, 0.0), 0.0, 1e-10);
    EXPECT_NEAR(struve_h(1, 0.0), 0.0, 1e-10);
    EXPECT_TRUE(std::isfinite(struve_h(0, 1.0)));
    EXPECT_TRUE(std::isfinite(struve_h(1, 1.0)));
}

TEST(SpecialGroup1, StruveHn_Smoke) {
    // struve_hn(int nu, double x) — verify finite output
    EXPECT_TRUE(std::isfinite(struve_hn(0, 1.0)));
    EXPECT_TRUE(std::isfinite(struve_hn(1, 1.0)));
    EXPECT_TRUE(std::isfinite(struve_hn(0, 2.0)));
    EXPECT_TRUE(std::isfinite(struve_hn(2, 1.5)));
}

TEST(SpecialGroup1, StruveL_Smoke) {
    // struve_l(int nu, double x) — verify finite output
    EXPECT_TRUE(std::isfinite(struve_l(0, 1.0)));
    EXPECT_TRUE(std::isfinite(struve_l(1, 1.0)));
    EXPECT_TRUE(std::isfinite(struve_l(0, 2.0)));
    EXPECT_TRUE(std::isfinite(struve_l(1, 0.5)));
}
