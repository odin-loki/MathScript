#pragma once

#include <functional>
#include <span>
#include <tuple>
#include <vector>

namespace ms {

// --- Descriptive ---
double mean(std::span<const double> data);
double var(std::span<const double> data);
double stddev(std::span<const double> data);
double median(std::span<const double> data);
double min_value(std::span<const double> data);
double max_value(std::span<const double> data);
double mode(std::span<const double> data);
double percentile(std::span<const double> data, double p);
double skewness(std::span<const double> data);
double kurtosis(std::span<const double> data);
double geometric_mean(std::span<const double> data);
double harmonic_mean(std::span<const double> data);
double rms(std::span<const double> data);
double mad(std::span<const double> data);     // median absolute deviation
double iqr(std::span<const double> data);     // interquartile range
double trimmed_mean(std::span<const double> data, double frac);

// --- Correlation ---
double correlation(std::span<const double> x, std::span<const double> y);
double spearman(std::span<const double> x, std::span<const double> y);
double kendall(std::span<const double> x, std::span<const double> y);

// --- Hypothesis tests ---
double ttest(std::span<const double> sample, double mu);
double two_sample_ttest(std::span<const double> a, std::span<const double> b);
double ztest(std::span<const double> sample, double mu, double sigma);
double chi2_gof(std::span<const double> observed, std::span<const double> expected);
double ks_test(std::span<const double> x,
               std::function<double(double)> cdf);

// One-way ANOVA (Analysis of Variance) across >= 2 groups. Returns the F-statistic and
// associated p-value testing the null hypothesis that all group means are equal.
struct AnovaResult {
    double f_stat;
    double p_value;
    int df_between;   // degrees of freedom between groups (k - 1)
    int df_within;    // degrees of freedom within groups (N - k)
};
AnovaResult one_way_anova(const std::vector<std::vector<double>>& groups);

// Mann-Whitney U test (a.k.a. Wilcoxon rank-sum test): a non-parametric test for whether two
// independent samples come from the same distribution, based on rank-sums rather than means
// (robust to non-normal distributions, unlike the t-test). Returns the U statistic and a
// normal-approximation p-value (standard for reasonably-sized samples; exact small-sample
// tables are not used here).
struct MannWhitneyResult {
    double u_stat;
    double p_value;
};
MannWhitneyResult mann_whitney_u(std::span<const double> a, std::span<const double> b);

// Kruskal-Wallis test: a non-parametric one-way ANOVA generalization for >= 2 groups, based on
// rank-sums across pooled samples (natural multi-group extension of the Mann-Whitney rank approach).
// Returns the H statistic (chi-square approximation with tie correction), degrees of freedom
// (number of non-empty groups - 1), and an upper-tail chi-squared p-value.
struct KruskalWallisResult {
    double h_stat = 0.0;
    int df = 0;
    double p_value = 1.0;
};
KruskalWallisResult kruskal_wallis(const std::vector<std::vector<double>>& groups);

// Friedman test: a non-parametric alternative to one-way repeated-measures ANOVA for >= 2
// treatments measured across >= 2 blocks (e.g. n judges each ranking k products). `data[i][j]`
// is the measurement for block i under treatment j, so `data` is an n_blocks x k_treatments
// matrix with every row the same length. Ranks the k_treatments values within each block
// (average ranks for ties), sums ranks per treatment, and forms a chi-square-distributed
// statistic (with tie correction) testing the null hypothesis that all treatments are
// equivalent. Returns the chi-squared statistic, degrees of freedom (k_treatments - 1), and an
// upper-tail chi-squared p-value. Malformed input (fewer than 2 blocks, fewer than 2
// treatments, or jagged/mismatched row lengths) yields a zeroed result with p_value == 1.0.
struct FriedmanResult {
    double chi2_stat = 0.0;
    int df = 0;
    double p_value = 1.0;
};
FriedmanResult friedman(const std::vector<std::vector<double>>& data);

// Two-sample Kolmogorov-Smirnov test: a non-parametric test for whether two samples come from
// the same distribution, based on the maximum distance between their empirical CDFs. Returns
// the D statistic and an asymptotic p-value (standard Kolmogorov distribution approximation).
struct KSTestResult {
    double d_stat;
    double p_value;
};
KSTestResult ks_test_2sample(std::span<const double> a, std::span<const double> b);

// Jarque-Bera test for normality: combines sample skewness and excess kurtosis into a single
// chi-squared statistic. Under the null hypothesis of normality, JB is asymptotically
// chi-squared distributed with 2 degrees of freedom. Returns the JB statistic, df (always 2),
// and an upper-tail chi-squared p-value.
struct JarqueBeraResult {
    double jb_stat = 0.0;
    int df = 2;
    double p_value = 1.0;
};
JarqueBeraResult jarque_bera(std::span<const double> x);

// Ljung-Box test for autocorrelation: tests whether a time series exhibits significant
// autocorrelation up to `max_lag`, using the sample ACF at lags 1..max_lag. Under the null
// hypothesis of no autocorrelation, Q is asymptotically chi-squared with `max_lag` degrees
// of freedom. Returns the Q statistic, df (= max_lag), and an upper-tail chi-squared p-value.
struct LjungBoxResult {
    double q_stat = 0.0;
    int df = 0;
    double p_value = 1.0;
};
LjungBoxResult ljung_box(std::span<const double> x, int max_lag);

// Levene's test for equality of variances across >= 2 groups (Brown-Forsythe median variant):
// transforms each group to absolute deviations from its median, then applies one-way ANOVA on
// the transformed values. Robust companion to ANOVA when checking the equal-variance assumption.
// Returns the F statistic, associated p-value, and ANOVA degrees of freedom (between/within).
struct LeveneResult {
    double f_stat = 0.0;
    double p_value = 0.0;
    int df_between = 0;
    int df_within = 0;
};
LeveneResult levene_test(const std::vector<std::vector<double>>& groups);

// Bartlett's test for equality of variances across >= 2 groups: assumes approximate normality
// within groups but is more powerful than Levene's when that holds. Uses a chi-squared
// approximation with Bartlett's correction factor. Returns the chi-squared statistic, df
// (number of non-empty groups - 1), and an upper-tail chi-squared p-value.
struct BartlettResult {
    double chi2_stat = 0.0;
    int df = 0;
    double p_value = 1.0;
};
BartlettResult bartlett_test(const std::vector<std::vector<double>>& groups);

// --- Regression ---
struct LinearRegressionResult {
    double slope;
    double intercept;
    double r_squared;
};
LinearRegressionResult linear_regression(std::span<const double> x,
                                          std::span<const double> y);
std::vector<double> multiple_regression(
    const std::vector<std::vector<double>>& X,
    std::span<const double> y);

// --- Time series ---
std::vector<double> acf(std::span<const double> x, int max_lag);
std::vector<double> pacf(std::span<const double> x, int max_lag);
std::vector<double> arfit(std::span<const double> x, int p); // AR(p) coefficients

// --- Bootstrap ---
double bootstrap_mean(std::span<const double> data, int n_boot = 1000,
                      unsigned seed = 42);
std::pair<double, double> bootstrap_ci(std::span<const double> data,
                                        double level = 0.95,
                                        int n_boot = 1000,
                                        unsigned seed = 42);

struct BootstrapResult {
    double point_estimate;   // the statistic computed on the original (full) sample
    double lower;             // lower bound of the confidence interval
    double upper;             // upper bound of the confidence interval
    double std_error;         // bootstrap standard error (std dev of the resampled statistic distribution)
};

// Computes a bootstrap confidence interval for an arbitrary statistic function `stat_fn`
// (e.g. mean, median, std dev, or any custom function of a data sample) via resampling with
// replacement. Uses the percentile method (the (alpha/2, 1-alpha/2) percentiles of the
// bootstrap distribution of the statistic) for the confidence interval — simple and standard,
// but with known limitations vs. BCa or studentized bootstrap for skewed distributions.
// `n_resamples` controls how many bootstrap samples to draw (default 1000); `confidence_level`
// e.g. 0.95 for a 95% CI; `seed` for reproducible resampling RNG.
BootstrapResult bootstrap_ci(
    std::span<const double> data,
    const std::function<double(std::span<const double>)>& stat_fn,
    size_t n_resamples = 1000,
    double confidence_level = 0.95,
    unsigned seed = 42);

} // namespace ms
