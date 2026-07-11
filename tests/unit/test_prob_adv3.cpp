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

TEST(ProbAdv3, Gamma_CDF_Exp_CrossCheck) {
    // Gamma(1, 1) = Exp(1): gamma_cdf(x, 1, 1) = 1 - exp(-x)
    for (double x : {0.5, 1.0, 2.0, 5.0}) {
        double g = gamma_cdf(x, 1.0, 1.0);
        double e = 1.0 - std::exp(-x);
        EXPECT_NEAR(g, e, 1e-6) << "Gamma(1,1) CDF should equal Exp(1) at x=" << x;
    }
}

TEST(ProbAdv3, Gamma_CDF_Monotone) {
    for (double shape : {1.0, 2.0, 5.0}) {
        double prev = gamma_cdf(0.0, shape, 1.0);
        for (double x : {0.5, 1.0, 2.0, 5.0, 10.0}) {
            double curr = gamma_cdf(x, shape, 1.0);
            EXPECT_GE(curr, prev - 1e-10) << "gamma_cdf not monotone at x=" << x;
            prev = curr;
        }
    }
}

TEST(ProbAdv3, Gamma_CDF_BoundaryAtZero) {
    EXPECT_NEAR(gamma_cdf(0.0, 2.0, 1.0), 0.0, 1e-10);
    EXPECT_NEAR(gamma_cdf(-1.0, 2.0, 1.0), 0.0, 1e-10);
}

TEST(ProbAdv3, Gamma_CDF_LargeX_NearOne) {
    EXPECT_GT(gamma_cdf(50.0, 2.0, 1.0), 0.999);
    EXPECT_GT(gamma_cdf(100.0, 5.0, 2.0), 0.999);
}

TEST(ProbAdv3, Gamma_CDF_InvalidParams) {
    EXPECT_DOUBLE_EQ(gamma_cdf(1.0, 0.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(gamma_cdf(1.0, 2.0, 0.0), 0.0);
}

// ---------------------------------------------------------------------------
// Beta distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv3, Beta_PDF_Uniform_CrossCheck) {
    // Beta(1, 1) = Uniform(0, 1): beta_pdf(x, 1, 1) = 1 for x in [0, 1]
    for (double x : {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}) {
        EXPECT_NEAR(beta_pdf(x, 1.0, 1.0), 1.0, 1e-10)
            << "Beta(1,1) PDF should equal 1 at x=" << x;
    }
}

TEST(ProbAdv3, Beta_CDF_Uniform_CrossCheck) {
    // Beta(1, 1) = Uniform(0, 1): beta_cdf(x, 1, 1) = x
    for (double x : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        EXPECT_NEAR(beta_cdf(x, 1.0, 1.0), x, 1e-10)
            << "Beta(1,1) CDF should equal x at x=" << x;
    }
}

TEST(ProbAdv3, Beta_CDF_Boundaries) {
    EXPECT_NEAR(beta_cdf(0.0, 2.0, 3.0), 0.0, 1e-10);
    EXPECT_NEAR(beta_cdf(1.0, 2.0, 3.0), 1.0, 1e-10);
}

TEST(ProbAdv3, Beta_PDF_IntegratesToOne) {
    auto integrate_beta = [](double alpha, double beta, int steps) {
        const double dx = 1.0 / static_cast<double>(steps);
        double sum = 0.0;
        for (int i = 0; i < steps; ++i) {
            const double x = (static_cast<double>(i) + 0.5) * dx;
            sum += beta_pdf(x, alpha, beta);
        }
        return sum * dx;
    };
    EXPECT_NEAR(integrate_beta(2.0, 3.0, 2000), 1.0, 1e-3);
    EXPECT_NEAR(integrate_beta(0.5, 0.5, 4000), 1.0, 1e-2);
}

TEST(ProbAdv3, Beta_PDF_OutsideSupport) {
    EXPECT_NEAR(beta_pdf(-0.1, 2.0, 2.0), 0.0, 1e-10);
    EXPECT_NEAR(beta_pdf(1.1, 2.0, 2.0), 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// F distribution
// ---------------------------------------------------------------------------

TEST(ProbAdv3, F_CDF_BoundaryAtZero) {
    EXPECT_NEAR(f_cdf(0.0, 5.0, 10.0), 0.0, 1e-10);
    EXPECT_NEAR(f_cdf(-1.0, 5.0, 10.0), 0.0, 1e-10);
}

TEST(ProbAdv3, F_CDF_LargeX_NearOne) {
    EXPECT_GT(f_cdf(100.0, 5.0, 10.0), 0.999);
    EXPECT_GT(f_cdf(50.0, 1.0, 30.0), 0.999);
}

TEST(ProbAdv3, F_CDF_T_Squared_CrossCheck) {
    // t^2 ~ F(1, df): f_cdf(t^2, 1, df) = 2*t_cdf(t, df) - 1 for t > 0
    for (double df : {5.0, 10.0, 30.0}) {
        for (double t_val : {0.5, 1.0, 2.0, 3.0}) {
            const double f_val = f_cdf(t_val * t_val, 1.0, df);
            const double t_val_cdf = 2.0 * t_cdf(t_val, df) - 1.0;
            EXPECT_NEAR(f_val, t_val_cdf, 1e-4)
                << "F(1," << df << ") CDF at t^2=" << t_val * t_val
                << " should match 2*t_cdf(" << t_val << ")-1";
        }
    }
}

TEST(ProbAdv3, F_PDF_NonNegative) {
    for (double d1 : {1.0, 5.0}) {
        for (double d2 : {5.0, 10.0}) {
            for (double x : {0.1, 0.5, 1.0, 2.0, 5.0}) {
                EXPECT_GE(f_pdf(x, d1, d2), 0.0);
                EXPECT_TRUE(std::isfinite(f_pdf(x, d1, d2)));
            }
        }
    }
}

TEST(ProbAdv3, F_PDF_IntegratesToOne) {
    auto integrate_f = [](double d1, double d2, double upper, int steps) {
        const double dx = upper / static_cast<double>(steps);
        double sum = 0.0;
        for (int i = 0; i < steps; ++i) {
            const double x = (static_cast<double>(i) + 0.5) * dx;
            sum += f_pdf(x, d1, d2);
        }
        return sum * dx;
    };
    EXPECT_NEAR(integrate_f(5.0, 10.0, 20.0, 4000), 1.0, 1e-2);
    EXPECT_NEAR(integrate_f(2.0, 8.0, 30.0, 6000), 1.0, 1e-2);
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

// ---------------------------------------------------------------------------
// Quantile functions (PPF) — Wave 183
// ---------------------------------------------------------------------------

TEST(ProbAdv3, Exp_PPF_ClosedForm) {
    for (double lambda : {0.5, 1.0, 2.5}) {
        for (double p : {0.1, 0.25, 0.5, 0.75, 0.9}) {
            const double expected = -std::log(1.0 - p) / lambda;
            EXPECT_NEAR(exp_ppf(p, lambda), expected, 1e-12)
                << "exp_ppf closed form at p=" << p << " lambda=" << lambda;
        }
    }
}

TEST(ProbAdv3, Exp_PPF_RoundTrip) {
    for (double lambda : {0.5, 1.0, 3.0}) {
        for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
            EXPECT_NEAR(exp_cdf(exp_ppf(p, lambda), lambda), p, 1e-6)
                << "exp round-trip at p=" << p << " lambda=" << lambda;
        }
    }
}

TEST(ProbAdv3, Chi2_PPF_TextbookCriticalValues) {
    EXPECT_NEAR(chi2_ppf(0.95, 1.0), 3.841, 1e-2);
    EXPECT_NEAR(chi2_ppf(0.95, 5.0), 11.070, 1e-2);
}

TEST(ProbAdv3, Chi2_PPF_RoundTrip) {
    for (double df : {1.0, 3.0, 5.0, 10.0, 30.0, 100.0}) {
        for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
            const double q = chi2_ppf(p, df);
            EXPECT_TRUE(std::isfinite(q));
            EXPECT_NEAR(chi2_cdf(q, df), p, 1e-6)
                << "chi2 round-trip at p=" << p << " df=" << df;
        }
    }
}

TEST(ProbAdv3, T_PPF_TextbookCriticalValues) {
    EXPECT_NEAR(t_ppf(0.975, 10.0), 2.228, 1e-2);
    EXPECT_NEAR(t_ppf(0.95, 30.0), 1.697, 1e-2);
}

TEST(ProbAdv3, T_PPF_RoundTrip) {
    for (double df : {1.0, 3.0, 10.0, 30.0, 100.0}) {
        for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
            const double q = t_ppf(p, df);
            EXPECT_TRUE(std::isfinite(q));
            EXPECT_NEAR(t_cdf(q, df), p, 1e-6)
                << "t round-trip at p=" << p << " df=" << df;
        }
    }
}

TEST(ProbAdv3, Gamma_PPF_RoundTrip) {
    for (double shape : {0.5, 1.0, 2.0, 5.0, 10.0}) {
        for (double scale : {0.5, 1.0, 2.0}) {
            for (double p : {0.01, 0.1, 0.5, 0.9, 0.99}) {
                const double q = gamma_ppf(p, shape, scale);
                EXPECT_TRUE(std::isfinite(q));
                EXPECT_NEAR(gamma_cdf(q, shape, scale), p, 1e-6)
                    << "gamma round-trip at p=" << p << " shape=" << shape
                    << " scale=" << scale;
            }
        }
    }
}

TEST(ProbAdv3, Gamma_PPF_ExpConsistency) {
    for (double scale : {0.5, 1.0, 2.0}) {
        for (double p : {0.1, 0.25, 0.5, 0.75, 0.9}) {
            EXPECT_NEAR(gamma_ppf(p, 1.0, scale), exp_ppf(p, 1.0 / scale), 1e-6)
                << "Gamma(1,scale) PPF should equal Exp PPF at p=" << p;
        }
    }
}

TEST(ProbAdv3, Beta_PPF_UniformConsistency) {
    for (double p : {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}) {
        if (p <= 0.0) {
            EXPECT_NEAR(beta_ppf(p, 1.0, 1.0), 0.0, 1e-10);
        } else if (p >= 1.0) {
            EXPECT_NEAR(beta_ppf(p, 1.0, 1.0), 1.0, 1e-10);
        } else {
            EXPECT_NEAR(beta_ppf(p, 1.0, 1.0), p, 1e-10)
                << "Beta(1,1) PPF should equal p at p=" << p;
        }
    }
}

TEST(ProbAdv3, Beta_PPF_RoundTrip) {
    for (double alpha : {0.5, 1.0, 2.0, 5.0}) {
        for (double beta_param : {0.5, 2.0, 5.0}) {
            for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
                const double q = beta_ppf(p, alpha, beta_param);
                EXPECT_TRUE(std::isfinite(q));
                EXPECT_GE(q, 0.0);
                EXPECT_LE(q, 1.0);
                EXPECT_NEAR(beta_cdf(q, alpha, beta_param), p, 1e-6)
                    << "beta round-trip at p=" << p << " alpha=" << alpha
                    << " beta=" << beta_param;
            }
        }
    }
}

TEST(ProbAdv3, F_PPF_RoundTrip) {
    for (double d1 : {1.0, 5.0, 10.0}) {
        for (double d2 : {5.0, 10.0, 30.0}) {
            for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
                const double q = f_ppf(p, d1, d2);
                EXPECT_TRUE(std::isfinite(q));
                EXPECT_NEAR(f_cdf(q, d1, d2), p, 1e-6)
                    << "f round-trip at p=" << p << " d1=" << d1 << " d2=" << d2;
            }
        }
    }
}

TEST(ProbAdv3, PPF_EdgeCases_NearBoundaries) {
    const double eps = 1e-10;

    EXPECT_NEAR(exp_ppf(0.0, 1.0), 0.0, 1e-10);
    EXPECT_GT(exp_ppf(1.0 - eps, 1.0), 10.0);

    EXPECT_LT(chi2_ppf(eps, 5.0), 0.01);
    EXPECT_GT(chi2_ppf(1.0 - eps, 5.0), 20.0);

    EXPECT_LT(t_ppf(eps, 10.0), -5.0);
    EXPECT_GT(t_ppf(1.0 - eps, 10.0), 5.0);
    EXPECT_NEAR(t_ppf(0.5, 10.0), 0.0, 1e-6);

    EXPECT_LT(gamma_ppf(eps, 2.0, 1.0), 0.01);
    EXPECT_GT(gamma_ppf(1.0 - eps, 2.0, 1.0), 10.0);

    EXPECT_LT(beta_ppf(eps, 2.0, 3.0), 0.01);
    EXPECT_NEAR(beta_ppf(1.0, 2.0, 3.0), 1.0, 1e-10);

    EXPECT_LT(f_ppf(eps, 5.0, 10.0), 0.01);
    EXPECT_GT(f_ppf(1.0 - eps, 5.0, 10.0), 10.0);
}

TEST(ProbAdv3, PPF_InvalidParams) {
    EXPECT_DOUBLE_EQ(exp_ppf(0.5, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(chi2_ppf(0.5, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(t_ppf(0.5, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(gamma_ppf(0.5, 0.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(gamma_ppf(0.5, 2.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(beta_ppf(0.5, 0.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(f_ppf(0.5, 0.0, 10.0), 0.0);
}
