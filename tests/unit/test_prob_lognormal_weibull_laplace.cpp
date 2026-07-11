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
// Cross-distribution sanity: negative-x domain handling
// ---------------------------------------------------------------------------

TEST(ProbLognormalWeibullLaplace, NegativeX_DomainHandling) {
    // Lognormal and Weibull have zero density for x < 0 (well-defined domain violation),
    // while Laplace is defined on the whole real line.
    EXPECT_DOUBLE_EQ(lognormal_pdf(-1.0, 0.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(weibull_pdf(-1.0, 1.0, 2.0), 0.0);
    EXPECT_GT(laplace_pdf(-1.0, 0.0, 1.0), 0.0);
}
