#include <gtest/gtest.h>
#include "ms/stats/stats.hpp"
#include "ms/prob/prob.hpp"

using namespace ms;

TEST(StatsExtTest, linear_regression) {
    std::vector<double> x{1, 2, 3, 4, 5};
    std::vector<double> y{2, 4, 6, 8, 10};
    auto fit = linear_regression(x, y);
    EXPECT_NEAR(fit.slope, 2.0, 1e-12);
    EXPECT_NEAR(fit.intercept, 0.0, 1e-12);
    EXPECT_NEAR(fit.r_squared, 1.0, 1e-12);
}

TEST(StatsExtTest, two_sample_ttest) {
    std::vector<double> a{1, 2, 3};
    std::vector<double> b{4, 5, 6};
    EXPECT_LT(two_sample_ttest(a, b), 0.0);
}

TEST(ProbExtTest, uniform_cdf) {
    EXPECT_NEAR(uniform_cdf(0.5, 0.0, 1.0), 0.5, 1e-12);
    EXPECT_NEAR(uniform_pdf(0.5, 0.0, 1.0), 1.0, 1e-12);
}

TEST(ProbExtTest, chi2_cdf) {
    EXPECT_NEAR(chi2_cdf(1.0, 1.0), 0.682689492, 1e-3);
}

TEST(ProbExtTest, gamma_pdf) {
    EXPECT_NEAR(gamma_pdf(1.0, 2.0, 1.0), 0.367879441, 1e-6);
}
