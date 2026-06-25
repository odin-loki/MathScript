// MathScript Advanced Probability Distribution Tests (Wave 54)
// Properties of t-distribution, chi-squared, gamma, uniform edge cases

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>

#include "ms/prob/prob.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// t-distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv3, T_PDF_Symmetric) {
    // t-distribution is symmetric: pdf(x) = pdf(-x)
    for (double df : {1.0, 3.0, 10.0}) {
        for (double x : {0.5, 1.0, 2.0}) {
            EXPECT_NEAR(t_pdf(x, df), t_pdf(-x, df), 1e-10)
                << "t_pdf not symmetric at x=" << x << " df=" << df;
        }
    }
}

TEST(ProbAdv3, T_PDF_MaxAtZero) {
    // t-pdf has maximum at x=0 for symmetric distribution
    for (double df : {1.0, 5.0, 30.0}) {
        double f0 = t_pdf(0.0, df);
        double f1 = t_pdf(1.0, df);
        EXPECT_GT(f0, f1) << "t_pdf(0) should be max, df=" << df;
    }
}

TEST(ProbAdv3, T_CDF_Half_At_Zero) {
    // By symmetry, CDF(0) = 0.5
    for (double df : {1.0, 3.0, 10.0, 30.0}) {
        EXPECT_NEAR(t_cdf(0.0, df), 0.5, 1e-6)
            << "t_cdf(0) should be 0.5 for df=" << df;
    }
}

TEST(ProbAdv3, T_CDF_Monotone) {
    for (double df : {2.0, 5.0}) {
        double prev = t_cdf(-5.0, df);
        for (double x : {-3.0, -1.0, 0.0, 1.0, 3.0}) {
            double curr = t_cdf(x, df);
            EXPECT_GE(curr, prev - 1e-10) << "t_cdf not monotone at x=" << x;
            prev = curr;
        }
    }
}

TEST(ProbAdv3, T_PDF_AllFinite_And_NonNeg) {
    for (double df : {1.0, 5.0, 30.0}) {
        for (double x : {-3.0, -1.0, 0.0, 1.0, 3.0}) {
            double p = t_pdf(x, df);
            EXPECT_TRUE(std::isfinite(p));
            EXPECT_GE(p, 0.0);
        }
    }
}

TEST(ProbAdv3, T_CDF_Bounds) {
    for (double df : {2.0, 10.0}) {
        EXPECT_LT(t_cdf(-10.0, df), 0.01) << "t_cdf(-10) should be near 0";
        EXPECT_GT(t_cdf(10.0, df), 0.99) << "t_cdf(10) should be near 1";
    }
}

// ---------------------------------------------------------------------------
// Chi-squared distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv3, Chi2_PDF_Zero_At_Zero) {
    // chi2_pdf(0, df) = 0 for df > 2
    EXPECT_NEAR(chi2_pdf(0.0, 3.0), 0.0, 1e-10);
    EXPECT_NEAR(chi2_pdf(0.0, 5.0), 0.0, 1e-10);
}

TEST(ProbAdv3, Chi2_PDF_Positive_ForPositiveX) {
    for (double df : {1.0, 2.0, 5.0}) {
        for (double x : {0.5, 1.0, 2.0, 5.0}) {
            EXPECT_GT(chi2_pdf(x, df), 0.0)
                << "chi2_pdf should be positive at x=" << x;
        }
    }
}

TEST(ProbAdv3, Chi2_CDF_Monotone) {
    for (double df : {2.0, 5.0}) {
        double prev = chi2_cdf(0.0, df);
        for (double x : {1.0, 2.0, 5.0, 10.0, 20.0}) {
            double curr = chi2_cdf(x, df);
            EXPECT_GE(curr, prev - 1e-10);
            prev = curr;
        }
    }
}

TEST(ProbAdv3, Chi2_CDF_AtZero_IsZero) {
    EXPECT_NEAR(chi2_cdf(0.0, 3.0), 0.0, 1e-6);
}

TEST(ProbAdv3, Chi2_CDF_LargeX_NearOne) {
    EXPECT_GT(chi2_cdf(100.0, 5.0), 0.999);
    EXPECT_GT(chi2_cdf(50.0, 2.0), 0.999);
}

// ---------------------------------------------------------------------------
// Gamma distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv3, Gamma_PDF_Positive_ForPositiveX) {
    for (double shape : {1.0, 2.0, 5.0}) {
        for (double x : {0.5, 1.0, 2.0}) {
            EXPECT_GT(gamma_pdf(x, shape, 1.0), 0.0);
        }
    }
}

TEST(ProbAdv3, Gamma_PDF_Exp_IsGamma_Shape1) {
    // Gamma(1, lambda) = Exp(lambda): gamma_pdf(x,1,1) = exp_pdf(x,1)
    for (double x : {0.5, 1.0, 2.0}) {
        double g = gamma_pdf(x, 1.0, 1.0);
        double e = std::exp(-x); // exp_pdf(x,1) = exp(-x)
        EXPECT_NEAR(g, e, 1e-6) << "Gamma(1,1) should equal Exp(1) at x=" << x;
    }
}

TEST(ProbAdv3, Gamma_PDF_AllFinite) {
    for (double shape : {0.5, 1.0, 2.0, 5.0}) {
        for (double x : {0.1, 0.5, 1.0, 2.0, 5.0}) {
            EXPECT_TRUE(std::isfinite(gamma_pdf(x, shape, 1.0)));
        }
    }
}

TEST(ProbAdv3, Gamma_PDF_Mode_AtShapeMinus1) {
    // Mode of Gamma(shape, 1) is at (shape-1) for shape >= 1
    // For shape=3: mode=2, pdf(2) > pdf(1) and pdf(2) > pdf(3)
    double p2 = gamma_pdf(2.0, 3.0, 1.0);
    double p1 = gamma_pdf(1.0, 3.0, 1.0);
    double p3 = gamma_pdf(3.0, 3.0, 1.0);
    EXPECT_GT(p2, p1) << "gamma_pdf(2,3,1) should be > gamma_pdf(1,3,1)";
    EXPECT_GT(p2, p3) << "gamma_pdf(2,3,1) should be > gamma_pdf(3,3,1)";
}

// ---------------------------------------------------------------------------
// Uniform distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv3, Uniform_PDF_Constant_In_Range) {
    // pdf(x, 0, 1) = 1 for x in (0,1)
    for (double x : {0.1, 0.3, 0.5, 0.7, 0.9}) {
        EXPECT_NEAR(uniform_pdf(x, 0.0, 1.0), 1.0, 1e-10);
    }
}

TEST(ProbAdv3, Uniform_PDF_Zero_Outside) {
    EXPECT_NEAR(uniform_pdf(-0.1, 0.0, 1.0), 0.0, 1e-10);
    EXPECT_NEAR(uniform_pdf(1.1, 0.0, 1.0), 0.0, 1e-10);
}

TEST(ProbAdv3, Uniform_CDF_Linear_In_Range) {
    // CDF(x, 0, 1) = x for x in [0,1]
    for (double x : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        EXPECT_NEAR(uniform_cdf(x, 0.0, 1.0), x, 1e-10);
    }
}

TEST(ProbAdv3, Uniform_CDF_Bounds) {
    EXPECT_NEAR(uniform_cdf(-1.0, 0.0, 1.0), 0.0, 1e-10);
    EXPECT_NEAR(uniform_cdf(2.0, 0.0, 1.0), 1.0, 1e-10);
}

TEST(ProbAdv3, Uniform_PDF_InverseWidth) {
    // pdf(x, a, b) = 1/(b-a)
    EXPECT_NEAR(uniform_pdf(2.0, 1.0, 3.0), 0.5, 1e-10);
    EXPECT_NEAR(uniform_pdf(5.0, 2.0, 7.0), 0.2, 1e-10);
}

// ---------------------------------------------------------------------------
// Normal distribution: additional properties
// ---------------------------------------------------------------------------

TEST(ProbAdv3, Normal_PDF_ScaleInvariant) {
    // Shifting data by mu doesn't change shape
    for (double x : {-1.0, 0.0, 1.0}) {
        double p0 = norm_pdf(x, 0.0, 1.0);
        double p1 = norm_pdf(x + 5.0, 5.0, 1.0); // shifted
        EXPECT_NEAR(p0, p1, 1e-10);
    }
}

TEST(ProbAdv3, Normal_CDF_Monotone) {
    double prev = norm_cdf(-5.0, 0.0, 1.0);
    for (double x : {-3.0, -1.0, 0.0, 1.0, 3.0}) {
        double curr = norm_cdf(x, 0.0, 1.0);
        EXPECT_GE(curr, prev - 1e-10);
        prev = curr;
    }
}
