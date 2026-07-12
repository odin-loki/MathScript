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

// Partial correlation between x and y after controlling for z:
// r_xy.z = (r_xy - r_xz*r_yz) / sqrt((1 - r_xz^2)*(1 - r_yz^2)).
double partial_correlation(std::span<const double> x,
                           std::span<const double> y,
                           std::span<const double> z);

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

// Fligner-Killeen test for equality of variances across >= 2 groups: a rank-based
// non-parametric alternative to Levene's and Bartlett's tests. Like Levene's median variant,
// each group is transformed to absolute deviations from its own median (robust to skewness in
// the raw data), but instead of running ANOVA on those deviations directly, the pooled
// deviations are converted to ranks (average ranks for ties) and then to expected normal order
// statistics (scores) via the standard normal quantile function `norm_ppf`. The Fligner-Killeen
// statistic is a weighted sum of squared deviations of each group's mean score from the overall
// mean score, normalized by the variance of the scores; under the null hypothesis of equal
// variances it is asymptotically chi-squared with (k - 1) degrees of freedom, where k is the
// number of non-empty groups. Because it operates on ranks/normal-scores rather than raw
// deviations, Fligner-Killeen is generally considered the most robust of the three variance
// tests here to non-normality and outliers/heavy tails, at some cost of power relative to
// Bartlett's test when the normality assumption actually holds. Returns the chi-squared
// statistic, df, and an upper-tail chi-squared p-value. Malformed input (fewer than 2 non-empty
// groups, or total sample size not exceeding the group count) yields a zeroed result with
// p_value == 1.0.
struct FlignerResult {
    double chi2_stat = 0.0;
    int df = 0;
    double p_value = 1.0;
};
FlignerResult fligner_test(const std::vector<std::vector<double>>& groups);

// Shapiro-Wilk test for normality: tests the null hypothesis that a sample was drawn from a
// normally distributed population, based on how closely the sample's order statistics
// correlate with the expected order statistics of a standard normal sample of the same size.
// Returns the W statistic (in (0, 1], with values close to 1 indicating consistency with
// normality and values well below 1 indicating departure from normality) and a p-value.
// Malformed/degenerate input (n < 3, or a constant sample with zero variance) yields the
// neutral default result { w_stat = 1.0, p_value = 1.0 }.
//
// @note Weights: uses the direct normalized-order-statistic weights
// a_i = m_i / sqrt(sum(m_j^2)), where m_i = norm_ppf((i - 0.375) / (n + 0.25)) approximates the
// expected order statistics of a standard normal sample (the standard Blom-type approximation),
// WITHOUT Royston's (1995) small-sample correction polynomial for the two most extreme weights
// (a_1, a_n). This is a simpler, still-valid variant of the textbook Royston algorithm
// (sometimes called the "Shapiro-Francia-style" simplification) that avoids needing to
// hardcode Royston's tail-correction coefficient tables; it is slightly less powerful than the
// full Royston weighting for small/moderate n but tracks it closely in practice.
// @note p-value: uses Royston's (1995) normalizing transformation, which maps a function of W
// to an approximately normally-distributed statistic via sample-size-dependent polynomial
// coefficients (one polynomial for 3 <= n <= 11, another for n > 11), then reports the
// upper-tail probability via norm_cdf. Because those coefficients were originally calibrated
// against Royston's tail-adjusted W rather than the simplified W computed here, and because the
// n <= 11 polynomial is nominally fitted for 4 <= n <= 11 (extended here down to n = 3 for
// simplicity rather than using the n = 3 exact closed form), the resulting p-value is an
// approximation: appropriate for statistical sanity-checking (monotonicity in W, clearly-normal
// vs. clearly-non-normal discrimination) but not for reproducing published p-value tables to
// high precision.
struct ShapiroWilkResult {
    double w_stat = 1.0;
    double p_value = 1.0;
};
ShapiroWilkResult shapiro_wilk(std::span<const double> x);

// Wilcoxon signed-rank test: a non-parametric test for whether the median of the paired
// differences (x[i] - y[i]) between two related/matched samples is zero (i.e. whether there's
// a systematic difference between paired observations), without assuming the differences are
// normally distributed (unlike a paired t-test). Pairs with a zero difference are discarded
// (see @note below); the remaining |differences| are ranked (average ranks for ties, via this
// module's `average_ranks` helper) and split into the rank-sum of positive differences (W+) and
// the rank-sum of negative differences (W-). Returns W = min(W+, W-), a normal-approximation z
// statistic, and a two-tailed p-value (standard for reasonably-sized samples; exact small-sample
// tables are not used here, consistent with `mann_whitney_u`/`kruskal_wallis` in this module).
//
// @note Zero-difference pairs: any pair with x[i] == y[i] contributes no information about the
// sign of a systematic difference and is excluded from ranking entirely, per the standard
// procedure for this test. `n_eff` is the number of pairs remaining after this exclusion.
// @note Tie correction: mirrors `mann_whitney_u`'s treatment in this module -- the variance of
// W is reduced by a tie-correction term derived from the sizes of tied groups among the ranked
// |differences|, matching the standard formula var_W = n(n+1)(2n+1)/24 - sum(t^3 - t)/48.
// @note Continuity correction: a 0.5 continuity correction is applied when converting W to a z
// statistic (consistent with `mann_whitney_u`'s use of a continuity correction).
// Malformed input (mismatched `x`/`y` lengths, or fewer than 1 non-zero-difference pair after
// discarding zero-difference pairs) yields the neutral default result
// { w_stat = 0.0, z_stat = 0.0, p_value = 1.0, n_eff = 0 }.
struct WilcoxonSignedRankResult {
    double w_stat = 0.0;   // W = min(W+, W-)
    double z_stat = 0.0;   // normal-approximation z statistic
    double p_value = 1.0;  // two-tailed p-value via normal approximation
    int n_eff = 0;          // number of non-zero-difference pairs actually used
};
WilcoxonSignedRankResult wilcoxon_signed_rank(std::span<const double> x, std::span<const double> y);

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

// Variance inflation factor for predictor column j of design matrix X (rows = observations,
// columns = predictors): regress column j on all other columns and return VIF_j = 1 / (1 - R^2).
// High values (commonly > 5 or > 10) indicate multicollinearity. Alias: vif().
double variance_inflation_factor(const std::vector<std::vector<double>>& X, size_t j);
double vif(const std::vector<std::vector<double>>& X, size_t j);

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
