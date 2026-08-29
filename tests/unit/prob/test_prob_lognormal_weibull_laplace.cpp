// MathScript: Lognormal, Weibull, Laplace Distribution Tests
// PDF/CDF/PPF trios for lognormal, Weibull, and Laplace distributions.

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <functional>

#include "ms/prob/prob.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

namespace {

// Simpson's rule over [a, b] with an even number of subintervals.
double simpson_integrate(const std::function<double(double)>& f, double a, double b, int steps) {
    if (steps % 2 != 0) {
        ++steps;
    }
    const double h = (b - a) / static_cast<double>(steps);
    double sum = f(a) + f(b);
    for (int i = 1; i < steps; ++i) {
        const double x = a + static_cast<double>(i) * h;
        sum += (i % 2 == 0 ? 2.0 : 4.0) * f(x);
    }
    return sum * h / 3.0;
}

} // namespace

// ---------------------------------------------------------------------------
// Lognormal distribution
// ---------------------------------------------------------------------------

TEST(ProbLognormal, PDF_Zero_ForNonPositiveX) {
    EXPECT_NEAR(lognormal_pdf(0.0, 0.0, 1.0), 0.0, 1e-12);
    EXPECT_NEAR(lognormal_pdf(-1.0, 0.0, 1.0), 0.0, 1e-12);
    EXPECT_NEAR(lognormal_pdf(-5.0, 1.0, 2.0), 0.0, 1e-12);
}

TEST(ProbLognormal, PDF_NonNegative_And_Finite) {
    for (double mu : {-1.0, 0.0, 1.0}) {
        for (double sigma : {0.25, 1.0, 2.0}) {
            for (double x : {-2.0, -0.5, 0.0, 0.1, 0.5, 1.0, 3.0, 10.0}) {
                const double p = lognormal_pdf(x, mu, sigma);
                EXPECT_GE(p, 0.0);
                EXPECT_TRUE(std::isfinite(p));
            }
        }
    }
}

TEST(ProbLognormal, PDF_InvalidSigma_IsZero) {
    EXPECT_DOUBLE_EQ(lognormal_pdf(1.0, 0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(lognormal_pdf(1.0, 0.0, -1.0), 0.0);
}

TEST(ProbLognormal, PDF_IntegratesToOne) {
    // Integrate via the substitution u = ln(x) (du = dx/x), i.e. integrate
    // g(u) = lognormal_pdf(exp(u), mu, sigma) * exp(u) over u. This keeps the
    // quadrature well-conditioned even for small sigma, where lognormal_pdf itself
    // is extremely peaked in x-space over a range spanning many orders of magnitude.
    for (double mu : {-1.0, 0.0, 0.5}) {
        for (double sigma : {0.25, 0.5, 1.0, 1.5}) {
            const auto g = [&](double u) { return lognormal_pdf(std::exp(u), mu, sigma) * std::exp(u); };
            const double area = simpson_integrate(g, mu - 10.0 * sigma, mu + 10.0 * sigma, 20000);
            EXPECT_NEAR(area, 1.0, 5e-3)
                << "lognormal_pdf should integrate to 1 for mu=" << mu << " sigma=" << sigma;
        }
    }
}

TEST(ProbLognormal, CDF_MatchesNumericalIntegral) {
    const double mu = 0.3, sigma = 0.7;
    const auto f = [&](double x) { return lognormal_pdf(x, mu, sigma); };
    for (double x : {0.2, 0.5, 1.0, 2.0, 5.0}) {
        const double integral = simpson_integrate(f, 1e-9, x, 20000);
        EXPECT_NEAR(lognormal_cdf(x, mu, sigma), integral, 5e-3)
            << "lognormal_cdf mismatch at x=" << x;
    }
}

TEST(ProbLognormal, CDF_Zero_ForNonPositiveX) {
    EXPECT_NEAR(lognormal_cdf(0.0, 0.0, 1.0), 0.0, 1e-12);
    EXPECT_NEAR(lognormal_cdf(-3.0, 0.0, 1.0), 0.0, 1e-12);
}

TEST(ProbLognormal, CDF_Monotone) {
    for (double sigma : {0.5, 1.0, 2.0}) {
        double prev = lognormal_cdf(0.01, 0.0, sigma);
        for (double x : {0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0}) {
            const double curr = lognormal_cdf(x, 0.0, sigma);
            EXPECT_GE(curr, prev - 1e-12);
            prev = curr;
        }
    }
}

TEST(ProbLognormal, PPF_RoundTrip_PPF_of_CDF) {
    for (double mu : {-1.0, 0.0, 1.0}) {
        for (double sigma : {0.3, 1.0, 2.0}) {
            for (double x : {0.1, 0.5, 1.0, 2.0, 10.0}) {
                const double p = lognormal_cdf(x, mu, sigma);
                // Skip the extreme tails: norm_cdf's 0.5*(1+erf(z)) form loses relative
                // precision there via catastrophic cancellation (independent of our
                // lognormal code), same reason existing norm_cdf tests only require
                // ~1e-5 absolute accuracy at |z|~10.
                if (p <= 1e-6 || p >= 1.0 - 1e-6) continue;
                const double x_rt = lognormal_ppf(p, mu, sigma);
                EXPECT_NEAR(x_rt, x, 1e-4 * std::max(1.0, x))
                    << "lognormal ppf(cdf(x)) != x at x=" << x << " mu=" << mu << " sigma=" << sigma;
            }
        }
    }
}

TEST(ProbLognormal, PPF_RoundTrip_CDF_of_PPF) {
    for (double mu : {-1.0, 0.0, 1.0}) {
        for (double sigma : {0.3, 1.0, 2.0}) {
            for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
                const double q = lognormal_ppf(p, mu, sigma);
                EXPECT_TRUE(std::isfinite(q));
                EXPECT_GT(q, 0.0);
                EXPECT_NEAR(lognormal_cdf(q, mu, sigma), p, 1e-6)
                    << "lognormal cdf(ppf(p)) != p at p=" << p << " mu=" << mu << " sigma=" << sigma;
            }
        }
    }
}

TEST(ProbLognormal, PPF_Monotone) {
    for (double sigma : {0.5, 1.5}) {
        double prev = lognormal_ppf(0.01, 0.0, sigma);
        for (double p : {0.05, 0.25, 0.5, 0.75, 0.95, 0.99}) {
            const double curr = lognormal_ppf(p, 0.0, sigma);
            EXPECT_GE(curr, prev - 1e-12);
            prev = curr;
        }
    }
}

TEST(ProbLognormal, PPF_MedianEqualsExpMu) {
    for (double mu : {-2.0, -0.5, 0.0, 0.5, 2.0}) {
        for (double sigma : {0.3, 1.0, 2.0}) {
            EXPECT_NEAR(lognormal_ppf(0.5, mu, sigma), std::exp(mu), 1e-6)
                << "lognormal median should equal exp(mu) for mu=" << mu;
        }
    }
}

TEST(ProbLognormal, PPF_Boundaries) {
    EXPECT_NEAR(lognormal_ppf(0.0, 0.0, 1.0), 0.0, 1e-12);
    EXPECT_GT(lognormal_ppf(0.999999, 0.0, 1.0), 100.0);
    EXPECT_DOUBLE_EQ(lognormal_ppf(0.5, 0.0, 0.0), 0.0);  // invalid sigma
}

// ---------------------------------------------------------------------------
// Weibull distribution
// ---------------------------------------------------------------------------

TEST(ProbWeibull, PDF_NonNegative_And_Finite) {
    for (double lambda : {0.5, 1.0, 3.0}) {
        for (double k : {0.5, 1.0, 2.0, 5.0}) {
            for (double x : {0.0, 0.1, 0.5, 1.0, 2.0, 5.0}) {
                const double p = weibull_pdf(x, lambda, k);
                EXPECT_GE(p, 0.0);
            }
        }
    }
}

TEST(ProbWeibull, PDF_Zero_ForNegativeX) {
    EXPECT_NEAR(weibull_pdf(-1.0, 1.0, 2.0), 0.0, 1e-12);
    EXPECT_NEAR(weibull_pdf(-0.1, 2.0, 0.5), 0.0, 1e-12);
}

TEST(ProbWeibull, PDF_InvalidParams_IsZero) {
    EXPECT_DOUBLE_EQ(weibull_pdf(1.0, 0.0, 2.0), 0.0);
    EXPECT_DOUBLE_EQ(weibull_pdf(1.0, -1.0, 2.0), 0.0);
    EXPECT_DOUBLE_EQ(weibull_pdf(1.0, 1.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(weibull_pdf(1.0, 1.0, -2.0), 0.0);
}

TEST(ProbWeibull, PDF_IntegratesToOne) {
    // Integrate via the substitution s = ln(x) (x = exp(s), dx = x ds), i.e. integrate
    // h(s) = weibull_pdf(exp(s), lambda, k) * exp(s) over s. Near x=0, h(s) ~ x^k -> 0,
    // so this removes the (integrable) singularity that weibull_pdf has at x=0 for k<1
    // and keeps the quadrature well-conditioned for every shape k.
    for (double lambda : {0.5, 1.0, 2.5}) {
        for (double k : {0.3, 0.7, 1.0, 2.0, 4.0}) {
            const auto h = [&](double s) { return weibull_pdf(std::exp(s), lambda, k) * std::exp(s); };
            const double hi = lambda * std::pow(30.0, 1.0 / k);
            // Lower bound chosen small enough that even for small k (slowly-decaying
            // near-zero tail) the truncated mass below exp(s_lo) is negligible.
            const double area = simpson_integrate(h, std::log(1e-30), std::log(hi), 20000);
            EXPECT_NEAR(area, 1.0, 5e-3)
                << "weibull_pdf should integrate to 1 for lambda=" << lambda << " k=" << k;
        }
    }
}

TEST(ProbWeibull, CDF_MatchesNumericalIntegral) {
    const double lambda = 1.5, k = 2.5;
    const auto f = [&](double x) { return weibull_pdf(x, lambda, k); };
    for (double x : {0.2, 0.5, 1.0, 2.0, 4.0}) {
        const double integral = simpson_integrate(f, 0.0, x, 20000);
        EXPECT_NEAR(weibull_cdf(x, lambda, k), integral, 5e-3)
            << "weibull_cdf mismatch at x=" << x;
    }
}

TEST(ProbWeibull, CDF_Zero_ForNegativeX_And_AtZero) {
    EXPECT_NEAR(weibull_cdf(-1.0, 1.0, 2.0), 0.0, 1e-12);
    EXPECT_NEAR(weibull_cdf(0.0, 1.0, 2.0), 0.0, 1e-12);
}

TEST(ProbWeibull, CDF_Monotone) {
    for (double k : {0.5, 1.0, 3.0}) {
        double prev = weibull_cdf(0.0, 1.0, k);
        for (double x : {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}) {
            const double curr = weibull_cdf(x, 1.0, k);
            EXPECT_GE(curr, prev - 1e-12);
            prev = curr;
        }
    }
}

TEST(ProbWeibull, CDF_AtScale_IsOneMinusInvE) {
    // Weibull CDF at x = lambda is always 1 - exp(-1), regardless of k.
    const double expected = 1.0 - std::exp(-1.0);
    for (double lambda : {0.5, 1.0, 2.0, 5.0}) {
        for (double k : {0.3, 0.7, 1.0, 2.0, 5.0}) {
            EXPECT_NEAR(weibull_cdf(lambda, lambda, k), expected, 1e-10)
                << "weibull_cdf(lambda, lambda, k) should be 1-exp(-1) for lambda=" << lambda
                << " k=" << k;
        }
    }
}

TEST(ProbWeibull, PPF_RoundTrip_PPF_of_CDF) {
    for (double lambda : {0.5, 1.0, 3.0}) {
        for (double k : {0.5, 1.0, 2.0, 4.0}) {
            for (double x : {0.1, 0.5, 1.0, 2.0, 5.0}) {
                const double p = weibull_cdf(x, lambda, k);
                if (p <= 0.0 || p >= 1.0) continue;
                const double x_rt = weibull_ppf(p, lambda, k);
                EXPECT_NEAR(x_rt, x, 1e-6 * std::max(1.0, x))
                    << "weibull ppf(cdf(x)) != x at x=" << x << " lambda=" << lambda << " k=" << k;
            }
        }
    }
}

TEST(ProbWeibull, PPF_RoundTrip_CDF_of_PPF) {
    for (double lambda : {0.5, 1.0, 3.0}) {
        for (double k : {0.5, 1.0, 2.0, 4.0}) {
            for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
                const double q = weibull_ppf(p, lambda, k);
                EXPECT_TRUE(std::isfinite(q));
                EXPECT_GE(q, 0.0);
                EXPECT_NEAR(weibull_cdf(q, lambda, k), p, 1e-6)
                    << "weibull cdf(ppf(p)) != p at p=" << p << " lambda=" << lambda << " k=" << k;
            }
        }
    }
}

TEST(ProbWeibull, PPF_Monotone) {
    for (double k : {0.5, 2.0}) {
        double prev = weibull_ppf(0.01, 1.0, k);
        for (double p : {0.05, 0.25, 0.5, 0.75, 0.95, 0.99}) {
            const double curr = weibull_ppf(p, 1.0, k);
            EXPECT_GE(curr, prev - 1e-12);
            prev = curr;
        }
    }
}

TEST(ProbWeibull, PPF_ClosedForm) {
    for (double lambda : {0.5, 1.0, 2.0}) {
        for (double k : {0.5, 1.0, 3.0}) {
            for (double p : {0.1, 0.3, 0.5, 0.7, 0.9}) {
                const double expected = lambda * std::pow(-std::log(1.0 - p), 1.0 / k);
                EXPECT_NEAR(weibull_ppf(p, lambda, k), expected, 1e-9)
                    << "weibull_ppf closed form at p=" << p;
            }
        }
    }
}

TEST(ProbWeibull, PPF_Boundaries) {
    EXPECT_NEAR(weibull_ppf(0.0, 1.0, 2.0), 0.0, 1e-12);
    EXPECT_GT(weibull_ppf(0.999999, 1.0, 2.0), 3.0);
    EXPECT_DOUBLE_EQ(weibull_ppf(0.5, 0.0, 2.0), 0.0);   // invalid lambda
    EXPECT_DOUBLE_EQ(weibull_ppf(0.5, 1.0, 0.0), 0.0);   // invalid k
}

TEST(ProbWeibull, K1_ReducesTo_Exponential) {
    // Weibull(lambda, 1) == Exponential(rate = 1/lambda)
    for (double lambda : {0.5, 1.0, 2.5}) {
        const double rate = 1.0 / lambda;
        for (double x : {0.0, 0.1, 0.5, 1.0, 2.0, 5.0}) {
            EXPECT_NEAR(weibull_pdf(x, lambda, 1.0), exp_pdf(x, rate), 1e-9)
                << "weibull_pdf(x,lambda,1) should equal exp_pdf(x,1/lambda) at x=" << x;
            EXPECT_NEAR(weibull_cdf(x, lambda, 1.0), exp_cdf(x, rate), 1e-9)
                << "weibull_cdf(x,lambda,1) should equal exp_cdf(x,1/lambda) at x=" << x;
        }
        for (double p : {0.1, 0.5, 0.9}) {
            EXPECT_NEAR(weibull_ppf(p, lambda, 1.0), exp_ppf(p, rate), 1e-9)
                << "weibull_ppf(p,lambda,1) should equal exp_ppf(p,1/lambda) at p=" << p;
        }
    }
}

// ---------------------------------------------------------------------------
// Laplace distribution
// ---------------------------------------------------------------------------

TEST(ProbLaplace, PDF_NonNegative_And_Finite) {
    for (double mu : {-2.0, 0.0, 3.0}) {
        for (double b : {0.5, 1.0, 2.0}) {
            for (double x : {-10.0, -1.0, 0.0, 1.0, 10.0}) {
                const double p = laplace_pdf(x, mu, b);
                EXPECT_GE(p, 0.0);
                EXPECT_TRUE(std::isfinite(p));
            }
        }
    }
}

TEST(ProbLaplace, PDF_InvalidScale_IsZero) {
    EXPECT_DOUBLE_EQ(laplace_pdf(0.0, 0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(laplace_pdf(0.0, 0.0, -1.0), 0.0);
}

TEST(ProbLaplace, PDF_Symmetric) {
    for (double mu : {-1.0, 0.0, 2.0}) {
        for (double b : {0.5, 1.0, 3.0}) {
            for (double d : {0.1, 0.5, 1.0, 3.0}) {
                EXPECT_NEAR(laplace_pdf(mu + d, mu, b), laplace_pdf(mu - d, mu, b), 1e-12)
                    << "laplace_pdf should be symmetric around mu=" << mu << " for d=" << d;
            }
        }
    }
}

TEST(ProbLaplace, PDF_IntegratesToOne) {
    for (double mu : {-1.0, 0.0, 2.0}) {
        for (double b : {0.5, 1.0, 2.0}) {
            const auto f = [&](double x) { return laplace_pdf(x, mu, b); };
            const double half_width = 30.0 * b;
            const double area = simpson_integrate(f, mu - half_width, mu + half_width, 20000);
            EXPECT_NEAR(area, 1.0, 5e-3)
                << "laplace_pdf should integrate to 1 for mu=" << mu << " b=" << b;
        }
    }
}

TEST(ProbLaplace, CDF_MatchesNumericalIntegral) {
    const double mu = 0.5, b = 1.2;
    const auto f = [&](double x) { return laplace_pdf(x, mu, b); };
    const double lo = mu - 30.0 * b;
    for (double x : {-2.0, 0.0, 0.5, 1.5, 3.0}) {
        const double integral = simpson_integrate(f, lo, x, 20000);
        EXPECT_NEAR(laplace_cdf(x, mu, b), integral, 5e-3)
            << "laplace_cdf mismatch at x=" << x;
    }
}

TEST(ProbLaplace, CDF_AtMu_IsExactlyHalf) {
    for (double mu : {-3.0, 0.0, 2.5}) {
        for (double b : {0.5, 1.0, 4.0}) {
            EXPECT_DOUBLE_EQ(laplace_cdf(mu, mu, b), 0.5);
        }
    }
}

TEST(ProbLaplace, CDF_Monotone) {
    for (double b : {0.5, 1.0, 2.0}) {
        double prev = laplace_cdf(-20.0, 0.0, b);
        for (double x : {-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0, 20.0}) {
            const double curr = laplace_cdf(x, 0.0, b);
            EXPECT_GE(curr, prev - 1e-12);
            prev = curr;
        }
    }
}

TEST(ProbLaplace, CDF_InvalidScale_IsZero) {
    EXPECT_DOUBLE_EQ(laplace_cdf(1.0, 0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(laplace_cdf(1.0, 0.0, -2.0), 0.0);
}

TEST(ProbLaplace, PPF_RoundTrip_PPF_of_CDF) {
    for (double mu : {-2.0, 0.0, 3.0}) {
        for (double b : {0.5, 1.0, 2.5}) {
            for (double x : {-5.0, -1.0, 0.0, 1.0, 5.0}) {
                const double p = laplace_cdf(x, mu, b);
                if (p <= 0.0 || p >= 1.0) continue;
                const double x_rt = laplace_ppf(p, mu, b);
                EXPECT_NEAR(x_rt, x, 1e-6 * std::max(1.0, std::abs(x)))
                    << "laplace ppf(cdf(x)) != x at x=" << x << " mu=" << mu << " b=" << b;
            }
        }
    }
}

TEST(ProbLaplace, PPF_RoundTrip_CDF_of_PPF) {
    for (double mu : {-2.0, 0.0, 3.0}) {
        for (double b : {0.5, 1.0, 2.5}) {
            for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
                const double q = laplace_ppf(p, mu, b);
                EXPECT_TRUE(std::isfinite(q));
                EXPECT_NEAR(laplace_cdf(q, mu, b), p, 1e-6)
                    << "laplace cdf(ppf(p)) != p at p=" << p << " mu=" << mu << " b=" << b;
            }
        }
    }
}

TEST(ProbLaplace, PPF_Monotone) {
    for (double b : {0.5, 2.0}) {
        double prev = laplace_ppf(0.01, 0.0, b);
        for (double p : {0.05, 0.25, 0.5, 0.75, 0.95, 0.99}) {
            const double curr = laplace_ppf(p, 0.0, b);
            EXPECT_GE(curr, prev - 1e-12);
            prev = curr;
        }
    }
}

TEST(ProbLaplace, PPF_ClosedForm_SignConvention) {
    // ppf(p) = mu - b*sign(p-0.5)*ln(1-2|p-0.5|)
    for (double mu : {-1.0, 0.0, 2.0}) {
        for (double b : {0.5, 1.0, 2.0}) {
            for (double p : {0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95}) {
                const double d = p - 0.5;
                const double sign = (d > 0.0) ? 1.0 : ((d < 0.0) ? -1.0 : 0.0);
                const double expected = mu - b * sign * std::log(1.0 - 2.0 * std::abs(d));
                EXPECT_NEAR(laplace_ppf(p, mu, b), expected, 1e-9)
                    << "laplace_ppf closed form mismatch at p=" << p;
            }
        }
    }
}

TEST(ProbLaplace, PPF_AtHalf_IsMu) {
    for (double mu : {-3.0, 0.0, 4.0}) {
        for (double b : {0.5, 1.0, 3.0}) {
            EXPECT_NEAR(laplace_ppf(0.5, mu, b), mu, 1e-10);
        }
    }
}

TEST(ProbLaplace, PPF_Boundaries) {
    EXPECT_LT(laplace_ppf(1e-10, 0.0, 1.0), -10.0);
    EXPECT_GT(laplace_ppf(1.0 - 1e-10, 0.0, 1.0), 10.0);
    EXPECT_DOUBLE_EQ(laplace_ppf(0.5, 0.0, 0.0), 0.0);   // invalid scale
}

// ---------------------------------------------------------------------------
// Logistic distribution
// ---------------------------------------------------------------------------

TEST(ProbLogistic, PDF_NonNegative_And_Finite) {
    for (double mu : {-2.0, 0.0, 3.0}) {
        for (double s : {0.5, 1.0, 2.0}) {
            for (double x : {-10.0, -1.0, 0.0, 1.0, 10.0}) {
                const double p = logistic_pdf(x, mu, s);
                EXPECT_GE(p, 0.0);
                EXPECT_TRUE(std::isfinite(p));
            }
        }
    }
}

TEST(ProbLogistic, PDF_InvalidScale_IsZero) {
    EXPECT_DOUBLE_EQ(logistic_pdf(0.0, 0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(logistic_pdf(0.0, 0.0, -1.0), 0.0);
}

TEST(ProbLogistic, PDF_Symmetric) {
    for (double mu : {-1.0, 0.0, 2.0}) {
        for (double s : {0.5, 1.0, 3.0}) {
            for (double d : {0.1, 0.5, 1.0, 3.0}) {
                EXPECT_NEAR(logistic_pdf(mu + d, mu, s), logistic_pdf(mu - d, mu, s), 1e-12)
                    << "logistic_pdf should be symmetric around mu=" << mu << " for d=" << d;
            }
        }
    }
}

TEST(ProbLogistic, PDF_IntegratesToOne) {
    for (double mu : {-1.0, 0.0, 2.0}) {
        for (double s : {0.5, 1.0, 2.0}) {
            const auto f = [&](double x) { return logistic_pdf(x, mu, s); };
            const double half_width = 30.0 * s;
            const double area = simpson_integrate(f, mu - half_width, mu + half_width, 20000);
            EXPECT_NEAR(area, 1.0, 5e-3)
                << "logistic_pdf should integrate to 1 for mu=" << mu << " s=" << s;
        }
    }
}

TEST(ProbLogistic, CDF_AtMu_IsExactlyHalf) {
    for (double mu : {-3.0, 0.0, 2.5}) {
        for (double s : {0.5, 1.0, 4.0}) {
            EXPECT_DOUBLE_EQ(logistic_cdf(mu, mu, s), 0.5);
        }
    }
}

TEST(ProbLogistic, CDF_Monotone) {
    for (double s : {0.5, 1.0, 2.0}) {
        double prev = logistic_cdf(-20.0, 0.0, s);
        for (double x : {-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0, 20.0}) {
            const double curr = logistic_cdf(x, 0.0, s);
            EXPECT_GE(curr, prev - 1e-12);
            prev = curr;
        }
    }
}

TEST(ProbLogistic, CDF_InvalidScale_IsZero) {
    EXPECT_DOUBLE_EQ(logistic_cdf(1.0, 0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(logistic_cdf(1.0, 0.0, -2.0), 0.0);
}

TEST(ProbLogistic, PPF_RoundTrip_PPF_of_CDF) {
    for (double mu : {-2.0, 0.0, 3.0}) {
        for (double s : {0.5, 1.0, 2.5}) {
            for (double x : {-5.0, -1.0, 0.0, 1.0, 5.0}) {
                const double p = logistic_cdf(x, mu, s);
                if (p <= 0.0 || p >= 1.0) continue;
                const double x_rt = logistic_ppf(p, mu, s);
                EXPECT_NEAR(x_rt, x, 1e-6 * std::max(1.0, std::abs(x)))
                    << "logistic ppf(cdf(x)) != x at x=" << x << " mu=" << mu << " s=" << s;
            }
        }
    }
}

TEST(ProbLogistic, PPF_RoundTrip_CDF_of_PPF) {
    for (double mu : {-2.0, 0.0, 3.0}) {
        for (double s : {0.5, 1.0, 2.5}) {
            for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
                const double q = logistic_ppf(p, mu, s);
                EXPECT_TRUE(std::isfinite(q));
                EXPECT_NEAR(logistic_cdf(q, mu, s), p, 1e-6)
                    << "logistic cdf(ppf(p)) != p at p=" << p << " mu=" << mu << " s=" << s;
            }
        }
    }
}

TEST(ProbLogistic, PPF_AtHalf_IsMu) {
    for (double mu : {-3.0, 0.0, 4.0}) {
        for (double s : {0.5, 1.0, 3.0}) {
            EXPECT_NEAR(logistic_ppf(0.5, mu, s), mu, 1e-10);
        }
    }
}

// ---------------------------------------------------------------------------
// Gumbel distribution
// ---------------------------------------------------------------------------

TEST(ProbGumbel, PDF_NonNegative_And_Finite) {
    for (double mu : {-2.0, 0.0, 3.0}) {
        for (double beta : {0.5, 1.0, 2.0}) {
            for (double x : {-10.0, -1.0, 0.0, 1.0, 10.0}) {
                const double p = gumbel_pdf(x, mu, beta);
                EXPECT_GE(p, 0.0);
                EXPECT_TRUE(std::isfinite(p));
            }
        }
    }
}

TEST(ProbGumbel, PDF_InvalidScale_IsZero) {
    EXPECT_DOUBLE_EQ(gumbel_pdf(0.0, 0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(gumbel_pdf(0.0, 0.0, -1.0), 0.0);
}

TEST(ProbGumbel, PDF_IntegratesToOne) {
    for (double mu : {-1.0, 0.0, 2.0}) {
        for (double beta : {0.5, 1.0, 2.0}) {
            const auto f = [&](double x) { return gumbel_pdf(x, mu, beta); };
            // Gumbel's left tail decays doubly-exponentially fast, so a modest
            // window around mu comfortably captures essentially all the mass.
            const double area = simpson_integrate(f, mu - 20.0 * beta, mu + 20.0 * beta, 20000);
            EXPECT_NEAR(area, 1.0, 5e-3)
                << "gumbel_pdf should integrate to 1 for mu=" << mu << " beta=" << beta;
        }
    }
}

TEST(ProbGumbel, CDF_MatchesNumericalIntegral) {
    const double mu = 0.4, beta = 1.3;
    const auto f = [&](double x) { return gumbel_pdf(x, mu, beta); };
    const double lo = mu - 20.0 * beta;
    for (double x : {-2.0, 0.0, 0.5, 1.5, 3.0}) {
        const double integral = simpson_integrate(f, lo, x, 20000);
        EXPECT_NEAR(gumbel_cdf(x, mu, beta), integral, 5e-3)
            << "gumbel_cdf mismatch at x=" << x;
    }
}

TEST(ProbGumbel, CDF_Monotone) {
    for (double beta : {0.5, 1.0, 2.0}) {
        double prev = gumbel_cdf(-20.0, 0.0, beta);
        for (double x : {-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0, 20.0}) {
            const double curr = gumbel_cdf(x, 0.0, beta);
            EXPECT_GE(curr, prev - 1e-12);
            prev = curr;
        }
    }
}

TEST(ProbGumbel, CDF_InvalidScale_IsZero) {
    EXPECT_DOUBLE_EQ(gumbel_cdf(1.0, 0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(gumbel_cdf(1.0, 0.0, -2.0), 0.0);
}

TEST(ProbGumbel, PPF_RoundTrip_PPF_of_CDF) {
    for (double mu : {-2.0, 0.0, 3.0}) {
        for (double beta : {0.5, 1.0, 2.5}) {
            for (double x : {-5.0, -1.0, 0.0, 1.0, 5.0}) {
                const double p = gumbel_cdf(x, mu, beta);
                if (p <= 0.0 || p >= 1.0) continue;
                const double x_rt = gumbel_ppf(p, mu, beta);
                EXPECT_NEAR(x_rt, x, 1e-6 * std::max(1.0, std::abs(x)))
                    << "gumbel ppf(cdf(x)) != x at x=" << x << " mu=" << mu << " beta=" << beta;
            }
        }
    }
}

TEST(ProbGumbel, PPF_RoundTrip_CDF_of_PPF) {
    for (double mu : {-2.0, 0.0, 3.0}) {
        for (double beta : {0.5, 1.0, 2.5}) {
            for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
                const double q = gumbel_ppf(p, mu, beta);
                EXPECT_TRUE(std::isfinite(q));
                EXPECT_NEAR(gumbel_cdf(q, mu, beta), p, 1e-6)
                    << "gumbel cdf(ppf(p)) != p at p=" << p << " mu=" << mu << " beta=" << beta;
            }
        }
    }
}

TEST(ProbGumbel, PPF_MedianClosedForm) {
    // Gumbel's median is mu - beta*log(log(2)).
    for (double mu : {-2.0, -0.5, 0.0, 0.5, 2.0}) {
        for (double beta : {0.3, 1.0, 2.0}) {
            const double expected = mu - beta * std::log(std::log(2.0));
            EXPECT_NEAR(gumbel_ppf(0.5, mu, beta), expected, 1e-9)
                << "gumbel median should equal mu - beta*log(log(2)) for mu=" << mu;
        }
    }
}

TEST(ProbGumbel, PPF_Boundaries) {
    // Gumbel's left tail decays doubly-exponentially, so even tiny-but-nonzero p only
    // maps to a modestly negative x; check the exact p=0/p=1 sentinel values instead.
    EXPECT_LT(gumbel_ppf(0.0, 0.0, 1.0), -1e50);
    EXPECT_GT(gumbel_ppf(1.0, 0.0, 1.0), 1e50);
    EXPECT_LT(gumbel_ppf(1e-10, 0.0, 1.0), 0.0);
    EXPECT_GT(gumbel_ppf(1.0 - 1e-10, 0.0, 1.0), 10.0);
    EXPECT_DOUBLE_EQ(gumbel_ppf(0.5, 0.0, 0.0), 0.0);   // invalid scale
}

// ---------------------------------------------------------------------------
// Cauchy distribution
// ---------------------------------------------------------------------------

TEST(ProbCauchy, PDF_NonNegative_And_Finite) {
    for (double x0 : {-2.0, 0.0, 3.0}) {
        for (double gamma : {0.5, 1.0, 2.0}) {
            for (double x : {-10.0, -1.0, 0.0, 1.0, 10.0}) {
                const double p = cauchy_pdf(x, x0, gamma);
                EXPECT_GE(p, 0.0);
                EXPECT_TRUE(std::isfinite(p));
            }
        }
    }
}

TEST(ProbCauchy, PDF_InvalidScale_IsZero) {
    EXPECT_DOUBLE_EQ(cauchy_pdf(0.0, 0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(cauchy_pdf(0.0, 0.0, -1.0), 0.0);
}

TEST(ProbCauchy, PDF_Symmetric) {
    for (double x0 : {-1.0, 0.0, 2.0}) {
        for (double gamma : {0.5, 1.0, 3.0}) {
            for (double d : {0.1, 0.5, 1.0, 3.0}) {
                EXPECT_NEAR(cauchy_pdf(x0 + d, x0, gamma), cauchy_pdf(x0 - d, x0, gamma), 1e-12)
                    << "cauchy_pdf should be symmetric around x0=" << x0 << " for d=" << d;
            }
        }
    }
}

TEST(ProbCauchy, PDF_IntegratesToOne) {
    // Cauchy's heavy tails decay only as 1/x^2, so the truncated-domain error is
    // O(1/window) rather than exponentially small; use a wide window accordingly.
    for (double x0 : {-1.0, 0.0, 2.0}) {
        for (double gamma : {0.5, 1.0, 2.0}) {
            const auto f = [&](double x) { return cauchy_pdf(x, x0, gamma); };
            const double half_width = 2000.0 * gamma;
            const double area = simpson_integrate(f, x0 - half_width, x0 + half_width, 40000);
            EXPECT_NEAR(area, 1.0, 5e-3)
                << "cauchy_pdf should integrate to 1 for x0=" << x0 << " gamma=" << gamma;
        }
    }
}

TEST(ProbCauchy, CDF_AtX0_IsExactlyHalf) {
    for (double x0 : {-3.0, 0.0, 2.5}) {
        for (double gamma : {0.5, 1.0, 4.0}) {
            EXPECT_DOUBLE_EQ(cauchy_cdf(x0, x0, gamma), 0.5);
        }
    }
}

TEST(ProbCauchy, CDF_Monotone) {
    for (double gamma : {0.5, 1.0, 2.0}) {
        double prev = cauchy_cdf(-1000.0, 0.0, gamma);
        for (double x : {-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0}) {
            const double curr = cauchy_cdf(x, 0.0, gamma);
            EXPECT_GE(curr, prev - 1e-12);
            prev = curr;
        }
    }
}

TEST(ProbCauchy, CDF_InvalidScale_IsZero) {
    EXPECT_DOUBLE_EQ(cauchy_cdf(1.0, 0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(cauchy_cdf(1.0, 0.0, -2.0), 0.0);
}

TEST(ProbCauchy, PPF_RoundTrip_PPF_of_CDF) {
    for (double x0 : {-2.0, 0.0, 3.0}) {
        for (double gamma : {0.5, 1.0, 2.5}) {
            for (double x : {-5.0, -1.0, 0.0, 1.0, 5.0}) {
                const double p = cauchy_cdf(x, x0, gamma);
                if (p <= 0.0 || p >= 1.0) continue;
                const double x_rt = cauchy_ppf(p, x0, gamma);
                EXPECT_NEAR(x_rt, x, 1e-6 * std::max(1.0, std::abs(x)))
                    << "cauchy ppf(cdf(x)) != x at x=" << x << " x0=" << x0 << " gamma=" << gamma;
            }
        }
    }
}

TEST(ProbCauchy, PPF_RoundTrip_CDF_of_PPF) {
    for (double x0 : {-2.0, 0.0, 3.0}) {
        for (double gamma : {0.5, 1.0, 2.5}) {
            for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
                const double q = cauchy_ppf(p, x0, gamma);
                EXPECT_TRUE(std::isfinite(q));
                EXPECT_NEAR(cauchy_cdf(q, x0, gamma), p, 1e-6)
                    << "cauchy cdf(ppf(p)) != p at p=" << p << " x0=" << x0 << " gamma=" << gamma;
            }
        }
    }
}

TEST(ProbCauchy, PPF_MedianAndModeEqualsX0) {
    for (double x0 : {-3.0, 0.0, 4.0}) {
        for (double gamma : {0.5, 1.0, 3.0}) {
            EXPECT_NEAR(cauchy_ppf(0.5, x0, gamma), x0, 1e-10);
        }
    }
}

TEST(ProbCauchy, PPF_Boundaries) {
    EXPECT_LT(cauchy_ppf(1e-10, 0.0, 1.0), -1000.0);
    EXPECT_GT(cauchy_ppf(1.0 - 1e-10, 0.0, 1.0), 1000.0);
    EXPECT_DOUBLE_EQ(cauchy_ppf(0.5, 0.0, 0.0), 0.0);   // invalid scale
}

// ---------------------------------------------------------------------------
// Pareto distribution
// ---------------------------------------------------------------------------

TEST(ProbPareto, PDF_NonNegative_And_Finite) {
    for (double x_m : {0.5, 1.0, 3.0}) {
        for (double alpha : {0.5, 1.0, 2.0, 5.0}) {
            for (double x : {x_m, x_m + 0.1, x_m + 1.0, x_m + 5.0, x_m * 10.0}) {
                const double p = pareto_pdf(x, x_m, alpha);
                EXPECT_GE(p, 0.0);
                EXPECT_TRUE(std::isfinite(p));
            }
        }
    }
}

TEST(ProbPareto, PDF_Zero_BelowSupport) {
    EXPECT_DOUBLE_EQ(pareto_pdf(0.5, 1.0, 2.0), 0.0);
    EXPECT_DOUBLE_EQ(pareto_pdf(0.0, 1.0, 2.0), 0.0);
    EXPECT_DOUBLE_EQ(pareto_pdf(-3.0, 1.0, 2.0), 0.0);
}

TEST(ProbPareto, PDF_InvalidParams_IsZero) {
    EXPECT_DOUBLE_EQ(pareto_pdf(1.0, 0.0, 2.0), 0.0);
    EXPECT_DOUBLE_EQ(pareto_pdf(1.0, -1.0, 2.0), 0.0);
    EXPECT_DOUBLE_EQ(pareto_pdf(1.0, 1.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(pareto_pdf(1.0, 1.0, -2.0), 0.0);
}

TEST(ProbPareto, PDF_Decreasing_AboveSupport) {
    for (double x_m : {0.5, 1.0, 2.0}) {
        for (double alpha : {0.5, 1.0, 3.0}) {
            double prev = pareto_pdf(x_m, x_m, alpha);
            for (double x : {x_m * 1.1, x_m * 1.5, x_m * 2.0, x_m * 5.0, x_m * 20.0}) {
                const double curr = pareto_pdf(x, x_m, alpha);
                EXPECT_LT(curr, prev)
                    << "pareto_pdf should strictly decrease for x > x_m at x=" << x;
                prev = curr;
            }
        }
    }
}

TEST(ProbPareto, PDF_IntegratesToOne) {
    // Integrate via the substitution u = ln(x/x_m) (x = x_m*exp(u), dx = x du), i.e.
    // integrate h(u) = pareto_pdf(x_m*exp(u), x_m, alpha) * x_m*exp(u) over u >= 0.
    // This keeps the quadrature well-conditioned for the power-law tail.
    for (double x_m : {0.5, 1.0, 2.5}) {
        for (double alpha : {0.5, 1.0, 2.0, 4.0}) {
            const auto h = [&](double u) {
                const double x = x_m * std::exp(u);
                return pareto_pdf(x, x_m, alpha) * x;
            };
            const double area = simpson_integrate(h, 0.0, 40.0, 40000);
            EXPECT_NEAR(area, 1.0, 5e-3)
                << "pareto_pdf should integrate to 1 for x_m=" << x_m << " alpha=" << alpha;
        }
    }
}

TEST(ProbPareto, CDF_Zero_AtAndBelowSupport) {
    EXPECT_DOUBLE_EQ(pareto_cdf(1.0, 1.0, 2.0), 0.0);
    EXPECT_DOUBLE_EQ(pareto_cdf(0.5, 1.0, 2.0), 0.0);
    EXPECT_DOUBLE_EQ(pareto_cdf(-1.0, 1.0, 2.0), 0.0);
}

TEST(ProbPareto, CDF_MatchesNumericalIntegral) {
    const double x_m = 1.5, alpha = 2.5;
    const auto f = [&](double x) { return pareto_pdf(x, x_m, alpha); };
    for (double x : {2.0, 3.0, 5.0, 10.0}) {
        const double integral = simpson_integrate(f, x_m, x, 20000);
        EXPECT_NEAR(pareto_cdf(x, x_m, alpha), integral, 5e-3)
            << "pareto_cdf mismatch at x=" << x;
    }
}

TEST(ProbPareto, CDF_Monotone) {
    for (double alpha : {0.5, 1.0, 3.0}) {
        double prev = pareto_cdf(1.0, 1.0, alpha);
        for (double x : {1.1, 1.5, 2.0, 5.0, 20.0}) {
            const double curr = pareto_cdf(x, 1.0, alpha);
            EXPECT_GE(curr, prev - 1e-12);
            prev = curr;
        }
    }
}

TEST(ProbPareto, CDF_InvalidParams_IsZero) {
    EXPECT_DOUBLE_EQ(pareto_cdf(2.0, 0.0, 2.0), 0.0);
    EXPECT_DOUBLE_EQ(pareto_cdf(2.0, -1.0, 2.0), 0.0);
    EXPECT_DOUBLE_EQ(pareto_cdf(2.0, 1.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(pareto_cdf(2.0, 1.0, -2.0), 0.0);
}

TEST(ProbPareto, PPF_RoundTrip_PPF_of_CDF) {
    for (double x_m : {0.5, 1.0, 3.0}) {
        for (double alpha : {0.5, 1.0, 2.0, 4.0}) {
            for (double x : {x_m + 0.1, x_m + 1.0, x_m * 2.0, x_m * 5.0}) {
                const double p = pareto_cdf(x, x_m, alpha);
                if (p <= 0.0 || p >= 1.0) continue;
                const double x_rt = pareto_ppf(p, x_m, alpha);
                EXPECT_NEAR(x_rt, x, 1e-6 * std::max(1.0, x))
                    << "pareto ppf(cdf(x)) != x at x=" << x << " x_m=" << x_m << " alpha=" << alpha;
            }
        }
    }
}

TEST(ProbPareto, PPF_RoundTrip_CDF_of_PPF) {
    for (double x_m : {0.5, 1.0, 3.0}) {
        for (double alpha : {0.5, 1.0, 2.0, 4.0}) {
            for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
                const double q = pareto_ppf(p, x_m, alpha);
                EXPECT_TRUE(std::isfinite(q));
                EXPECT_GE(q, x_m);
                EXPECT_NEAR(pareto_cdf(q, x_m, alpha), p, 1e-6)
                    << "pareto cdf(ppf(p)) != p at p=" << p << " x_m=" << x_m << " alpha=" << alpha;
            }
        }
    }
}

TEST(ProbPareto, PPF_AtZero_IsXm) {
    for (double x_m : {0.5, 1.0, 4.0}) {
        for (double alpha : {0.5, 1.0, 3.0}) {
            EXPECT_NEAR(pareto_ppf(0.0, x_m, alpha), x_m, 1e-10);
        }
    }
}

TEST(ProbPareto, PPF_Boundaries) {
    EXPECT_GT(pareto_ppf(1.0 - 1e-10, 1.0, 2.0), 1000.0);
    EXPECT_DOUBLE_EQ(pareto_ppf(0.5, 0.0, 2.0), 0.0);   // invalid x_m
    EXPECT_DOUBLE_EQ(pareto_ppf(0.5, 1.0, 0.0), 0.0);   // invalid alpha
}

// ---------------------------------------------------------------------------
// Rayleigh distribution
// ---------------------------------------------------------------------------

TEST(ProbRayleigh, PDF_NonNegative_And_Finite) {
    for (double sigma : {0.5, 1.0, 3.0}) {
        for (double x : {0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0}) {
            const double p = rayleigh_pdf(x, sigma);
            EXPECT_GE(p, 0.0);
            EXPECT_TRUE(std::isfinite(p));
        }
    }
}

TEST(ProbRayleigh, PDF_Zero_ForNegativeX) {
    EXPECT_DOUBLE_EQ(rayleigh_pdf(-1.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(rayleigh_pdf(-0.1, 2.5), 0.0);
}

TEST(ProbRayleigh, PDF_InvalidSigma_IsZero) {
    EXPECT_DOUBLE_EQ(rayleigh_pdf(1.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(rayleigh_pdf(1.0, -1.0), 0.0);
}

TEST(ProbRayleigh, PDF_IntegratesToOne) {
    for (double sigma : {0.5, 1.0, 2.0, 3.5}) {
        const auto f = [&](double x) { return rayleigh_pdf(x, sigma); };
        const double area = simpson_integrate(f, 0.0, 10.0 * sigma, 20000);
        EXPECT_NEAR(area, 1.0, 5e-3)
            << "rayleigh_pdf should integrate to 1 for sigma=" << sigma;
    }
}

TEST(ProbRayleigh, PDF_ModeIsAtSigma) {
    // The Rayleigh mode (pdf maximum) is located at x = sigma.
    for (double sigma : {0.5, 1.0, 2.0, 4.0}) {
        const double eps = 1e-3 * sigma;
        const double at_mode = rayleigh_pdf(sigma, sigma);
        const double below = rayleigh_pdf(sigma - eps, sigma);
        const double above = rayleigh_pdf(sigma + eps, sigma);
        EXPECT_LT(below, at_mode) << "rayleigh_pdf should increase up to the mode at sigma=" << sigma;
        EXPECT_GT(at_mode, above) << "rayleigh_pdf should decrease past the mode at sigma=" << sigma;
    }
}

TEST(ProbRayleigh, CDF_MatchesNumericalIntegral) {
    const double sigma = 1.7;
    const auto f = [&](double x) { return rayleigh_pdf(x, sigma); };
    for (double x : {0.2, 0.5, 1.0, 2.0, 5.0}) {
        const double integral = simpson_integrate(f, 0.0, x, 20000);
        EXPECT_NEAR(rayleigh_cdf(x, sigma), integral, 5e-3)
            << "rayleigh_cdf mismatch at x=" << x;
    }
}

TEST(ProbRayleigh, CDF_Zero_AtZero_And_ApproachesOne) {
    for (double sigma : {0.5, 1.0, 3.0}) {
        EXPECT_DOUBLE_EQ(rayleigh_cdf(0.0, sigma), 0.0);
        EXPECT_NEAR(rayleigh_cdf(20.0 * sigma, sigma), 1.0, 1e-12);
    }
}

TEST(ProbRayleigh, CDF_Zero_ForNegativeX) {
    EXPECT_DOUBLE_EQ(rayleigh_cdf(-1.0, 1.0), 0.0);
}

TEST(ProbRayleigh, CDF_InvalidSigma_IsZero) {
    EXPECT_DOUBLE_EQ(rayleigh_cdf(1.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(rayleigh_cdf(1.0, -2.0), 0.0);
}

TEST(ProbRayleigh, CDF_Monotone) {
    for (double sigma : {0.5, 1.0, 2.0}) {
        double prev = rayleigh_cdf(0.0, sigma);
        for (double x : {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}) {
            const double curr = rayleigh_cdf(x, sigma);
            EXPECT_GE(curr, prev - 1e-12);
            prev = curr;
        }
    }
}

TEST(ProbRayleigh, CDF_AtSigma_ClosedForm) {
    // rayleigh_cdf(sigma, sigma) == 1 - exp(-0.5) regardless of sigma.
    const double expected = 1.0 - std::exp(-0.5);
    for (double sigma : {0.3, 1.0, 2.5, 7.0}) {
        EXPECT_NEAR(rayleigh_cdf(sigma, sigma), expected, 1e-10)
            << "rayleigh_cdf(sigma, sigma) should be 1-exp(-0.5) for sigma=" << sigma;
    }
}

TEST(ProbRayleigh, PPF_RoundTrip_PPF_of_CDF) {
    for (double sigma : {0.5, 1.0, 3.0}) {
        for (double x : {0.1, 0.5, 1.0, 2.0, 5.0}) {
            const double p = rayleigh_cdf(x, sigma);
            if (p <= 0.0 || p >= 1.0) continue;
            const double x_rt = rayleigh_ppf(p, sigma);
            EXPECT_NEAR(x_rt, x, 1e-6 * std::max(1.0, x))
                << "rayleigh ppf(cdf(x)) != x at x=" << x << " sigma=" << sigma;
        }
    }
}

TEST(ProbRayleigh, PPF_RoundTrip_CDF_of_PPF) {
    for (double sigma : {0.5, 1.0, 3.0}) {
        for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
            const double q = rayleigh_ppf(p, sigma);
            EXPECT_TRUE(std::isfinite(q));
            EXPECT_GE(q, 0.0);
            EXPECT_NEAR(rayleigh_cdf(q, sigma), p, 1e-6)
                << "rayleigh cdf(ppf(p)) != p at p=" << p << " sigma=" << sigma;
        }
    }
}

TEST(ProbRayleigh, PPF_Monotone) {
    for (double sigma : {0.5, 2.0}) {
        double prev = rayleigh_ppf(0.01, sigma);
        for (double p : {0.05, 0.25, 0.5, 0.75, 0.95, 0.99}) {
            const double curr = rayleigh_ppf(p, sigma);
            EXPECT_GE(curr, prev - 1e-12);
            prev = curr;
        }
    }
}

TEST(ProbRayleigh, PPF_ClosedForm) {
    for (double sigma : {0.5, 1.0, 2.0}) {
        for (double p : {0.1, 0.3, 0.5, 0.7, 0.9}) {
            const double expected = sigma * std::sqrt(-2.0 * std::log(1.0 - p));
            EXPECT_NEAR(rayleigh_ppf(p, sigma), expected, 1e-9)
                << "rayleigh_ppf closed form mismatch at p=" << p;
        }
    }
}

TEST(ProbRayleigh, PPF_Boundaries) {
    EXPECT_DOUBLE_EQ(rayleigh_ppf(0.0, 1.0), 0.0);
    EXPECT_GT(rayleigh_ppf(0.999999, 1.0), 3.0);
    EXPECT_DOUBLE_EQ(rayleigh_ppf(0.5, 0.0), 0.0);   // invalid sigma
    EXPECT_DOUBLE_EQ(rayleigh_ppf(0.5, -1.0), 0.0);  // invalid sigma
}

TEST(ProbRayleigh, EquivalentToWeibull_K2) {
    // Rayleigh(sigma) == Weibull(lambda=sigma*sqrt(2), k=2).
    for (double sigma : {0.5, 1.0, 2.5}) {
        const double lambda = sigma * std::sqrt(2.0);
        for (double x : {0.0, 0.1, 0.5, 1.0, 2.0, 5.0}) {
            EXPECT_NEAR(rayleigh_pdf(x, sigma), weibull_pdf(x, lambda, 2.0), 1e-9)
                << "rayleigh_pdf(x,sigma) should equal weibull_pdf(x,sigma*sqrt(2),2) at x=" << x;
            EXPECT_NEAR(rayleigh_cdf(x, sigma), weibull_cdf(x, lambda, 2.0), 1e-9)
                << "rayleigh_cdf(x,sigma) should equal weibull_cdf(x,sigma*sqrt(2),2) at x=" << x;
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-distribution sanity: negative-x domain handling
// ---------------------------------------------------------------------------

TEST(ProbLognormalWeibullLaplace, NegativeX_DomainHandling) {
    // Lognormal and Weibull have zero density for x < 0 (well-defined domain violation),
    // while Laplace is defined on the whole real line.
    EXPECT_DOUBLE_EQ(lognormal_pdf(-1.0, 0.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(weibull_pdf(-1.0, 1.0, 2.0), 0.0);
    EXPECT_GT(laplace_pdf(-1.0, 0.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(rayleigh_pdf(-1.0, 1.0), 0.0);
}
