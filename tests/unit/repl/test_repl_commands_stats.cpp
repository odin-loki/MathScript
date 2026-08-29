#include <algorithm>
#include <cmath>
#include <set>
#include <fstream>
#include <gtest/gtest.h>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ms/cplx/cplx.hpp"
#include "ms/control/control.hpp"
#include "ms/error/error_types.hpp"
#include "ms/finance/finance.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/ml/ml.hpp"
#include "ms/pde/pde.hpp"
#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/version.hpp"

#include "repl/repl_test_helpers.hpp"

using namespace ms::interp;

TEST(ReplCommandsTest, stats_correlation) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_correlation(x,y)");

    expect_ok(interp, "r = stats_correlation([1; 2; 3; 4; 5], [2; 4; 6; 8; 10])");
    EXPECT_NEAR(interp.state().scalars.at("r"), 1.0, 1e-9);
}

TEST(ReplCommandsTest, stats_mean) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_mean(x)");

    expect_ok(interp, "m = stats_mean([1; 2; 3; 4; 5])");
    EXPECT_NEAR(interp.state().scalars.at("m"), 3.0, 1e-9);

    expect_contains(interp, "stats_mean([1; 2; 3; 4; 5])", "3");
}

TEST(ReplCommandsTest, stats_spearman) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_spearman(x,y)");

    expect_ok(interp, "sp = stats_spearman([1; 2; 3; 4; 5], [2; 4; 6; 8; 10])");
    EXPECT_NEAR(interp.state().scalars.at("sp"), 1.0, 1e-9);

    expect_contains(interp, "stats_spearman([1; 2; 3; 4; 5], [2; 4; 6; 8; 10])", "1");
}

TEST(ReplCommandsTest, stats_median) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_median(x)");

    expect_ok(interp, "med = stats_median([1; 2; 3; 4; 5])");
    EXPECT_NEAR(interp.state().scalars.at("med"), 3.0, 1e-9);

    expect_contains(interp, "stats_median([1; 2; 3; 4; 5])", "3");
}

TEST(ReplCommandsTest, stats_kendall) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_kendall(x,y)");

    expect_ok(interp, "kt = stats_kendall([1; 2; 3; 4; 5], [2; 4; 6; 8; 10])");
    EXPECT_NEAR(interp.state().scalars.at("kt"), 1.0, 1e-9);

    expect_contains(interp, "stats_kendall([1; 2; 3; 4; 5], [2; 4; 6; 8; 10])", "1");
}

TEST(ReplCommandsTest, stats_stddev) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_stddev(x)");

    expect_ok(interp, "sd = stats_stddev([1; 2; 3; 4; 5])");
    EXPECT_NEAR(interp.state().scalars.at("sd"), std::sqrt(2.5), 1e-9);

    expect_contains(interp, "stats_stddev([1; 2; 3; 4; 5])", "1.58114");
}

TEST(ReplCommandsTest, stats_skewness) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_skewness(x)");

    expect_ok(interp, "sk = stats_skewness([1; 2; 3; 4; 5])");
    EXPECT_NEAR(interp.state().scalars.at("sk"), 0.0, 1e-9);

    expect_contains(interp, "stats_skewness([1; 2; 3; 4; 5])", "0");
}

TEST(ReplCommandsTest, stats_kurtosis) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_kurtosis(x)");

    expect_ok(interp, "ku = stats_kurtosis([1; 2; 3; 4; 5])");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ku")));
}

TEST(ReplCommandsTest, stats_var) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_var(x)");

    expect_ok(interp, "v = stats_var([1; 2; 3; 4; 5])");
    EXPECT_NEAR(interp.state().scalars.at("v"), 2.5, 1e-9);

    expect_contains(interp, "stats_var([1; 2; 3; 4; 5])", "2.5");
}

TEST(ReplCommandsTest, stats_percentile) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_percentile(x,p)");

    expect_ok(interp, "p50 = stats_percentile([1; 2; 3; 4; 5], 50)");
    EXPECT_NEAR(interp.state().scalars.at("p50"), 3.0, 1e-9);

    expect_contains(interp, "stats_percentile([1; 2; 3; 4; 5], 50)", "3");
}

TEST(ReplCommandsTest, stats_mode) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_mode(x)");

    expect_ok(interp, "m = stats_mode([1; 2; 2; 3])");
    EXPECT_NEAR(interp.state().scalars.at("m"), 2.0, 1e-9);

    expect_contains(interp, "stats_mode([1; 2; 2; 3])", "2");
}

TEST(ReplCommandsTest, stats_geometric_mean) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_geometric_mean(x)");

    expect_ok(interp, "gm = stats_geometric_mean([2; 8])");
    EXPECT_NEAR(interp.state().scalars.at("gm"), 4.0, 1e-9);

    expect_contains(interp, "stats_geometric_mean([2; 8])", "4");
}

TEST(ReplCommandsTest, stats_ttest) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_ttest(x,mu)");

    expect_ok(interp, "t = stats_ttest([6; 7; 8; 9; 10], 5)");
    EXPECT_GT(interp.state().scalars.at("t"), 0.0);

    expect_ok(interp, "stats_ttest([6; 7; 8; 9; 10], 5)");
}

TEST(ReplCommandsTest, stats_harmonic_mean) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_harmonic_mean(x)");

    expect_ok(interp, "hm = stats_harmonic_mean([1; 2; 3; 4])");
    EXPECT_NEAR(interp.state().scalars.at("hm"), 48.0 / 25.0, 1e-9);

    expect_contains(interp, "stats_harmonic_mean([1; 2; 3; 4])", "1.92");
}

TEST(ReplCommandsTest, stats_rms) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_rms(x)");

    expect_ok(interp, "r = stats_rms([3; 4])");
    EXPECT_NEAR(interp.state().scalars.at("r"), std::sqrt(12.5), 1e-9);

    expect_contains(interp, "stats_rms([3; 4])", "3.5355");
}

TEST(ReplCommandsTest, stats_mad) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_mad(x)");

    expect_ok(interp, "m = stats_mad([1; 1; 2; 2; 4; 6; 9])");
    EXPECT_NEAR(interp.state().scalars.at("m"), 1.4826, 1e-9);

    expect_contains(interp, "stats_mad([1; 1; 2; 2; 4; 6; 9])", "1.4826");
}

TEST(ReplCommandsTest, stats_ztest) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_ztest(x,mu,sigma)");

    expect_ok(interp, "z = stats_ztest([6; 7; 8; 9; 10], 5, 1)");
    EXPECT_GT(interp.state().scalars.at("z"), 0.0);

    expect_ok(interp, "stats_ztest([6; 7; 8; 9; 10], 5, 1)");
}

TEST(ReplCommandsTest, stats_acf) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_acf(x,max_lag)");

    expect_ok(interp, "a = stats_acf([1; 2; 3; 4; 5], 2)");
    ASSERT_GT(interp.state().matrices.count("a"), 0u);
    EXPECT_EQ(interp.state().matrices.at("a").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("a")(0, 0), 1.0, 1e-9);

    expect_ok(interp, "stats_acf([1; 2; 3; 4; 5], 2)");
}

TEST(ReplCommandsTest, stats_iqr) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_iqr(x)");

    expect_ok(interp, "iq = stats_iqr([1; 2; 3; 4; 5; 6; 7; 8; 9])");
    EXPECT_GT(interp.state().scalars.at("iq"), 0.0);

    expect_contains(interp, "stats_iqr([1; 2; 3; 4; 5; 6; 7; 8; 9])", "4");
}

TEST(ReplCommandsTest, stats_two_sample_ttest) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_two_sample_ttest(a,b)");

    expect_ok(interp, "t2 = stats_two_sample_ttest([1; 2; 3], [4; 5; 6])");
    EXPECT_LT(interp.state().scalars.at("t2"), 0.0);

    expect_ok(interp, "stats_two_sample_ttest([1; 2; 3], [4; 5; 6])");
}

TEST(ReplCommandsTest, stats_chi2_gof) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_chi2_gof(observed,expected)");

    expect_ok(interp, "g = stats_chi2_gof([10; 20; 30], [10; 20; 30])");
    EXPECT_NEAR(interp.state().scalars.at("g"), 0.0, 1e-9);

    expect_ok(interp, "stats_chi2_gof([10; 20; 30], [10; 20; 30])");
}

TEST(ReplCommandsTest, stats_min_value) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_min_value(x)");

    expect_ok(interp, "m = stats_min_value([3; 1; 4; 1; 5; 9; 2])");
    EXPECT_NEAR(interp.state().scalars.at("m"), 1.0, 1e-9);
    expect_contains(interp, "stats_min_value([3; 1; 4; 1; 5; 9; 2])", "1");
}

TEST(ReplCommandsTest, stats_max_value) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_max_value(x)");

    expect_ok(interp, "m = stats_max_value([3; 1; 4; 1; 5; 9; 2])");
    EXPECT_NEAR(interp.state().scalars.at("m"), 9.0, 1e-9);
    expect_contains(interp, "stats_max_value([3; 1; 4; 1; 5; 9; 2])", "9");
}

TEST(ReplCommandsTest, stats_ts) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_linear_regression(x,y)");
    expect_contains(interp, "help", "stats_pacf(x,max_lag)");
    expect_contains(interp, "help", "stats_kde(samples,grid,h[,kernel])");
    expect_contains(interp, "help", "stats_bootstrap_ci(x)");

    // y = 2x + 1
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    ASSERT_GT(interp.state().matrices.count("lr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lr").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("lr").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 2), 1.0, 1e-9);

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("p").cols(), 1u);

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    ASSERT_GT(interp.state().matrices.count("k"), 0u);
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("k").cols(), 1u);
    for (size_t i = 0; i < 7; ++i) {
        EXPECT_TRUE(std::isfinite(interp.state().matrices.at("k")(i, 0)));
    }

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    ASSERT_GT(interp.state().matrices.count("ci"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ci").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("ci")(0, 0), 5.5, 1e-9);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("ci")(0, 1)));
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("ci")(0, 2)));
    EXPECT_GT(interp.state().matrices.at("ci")(0, 3), 0.0);
}

TEST(ReplCommandsTest, stats_inference) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_shapiro_wilk(x)");
    expect_contains(interp, "help", "stats_mann_whitney_u(a,b)");
    expect_contains(interp, "help", "stats_one_way_anova(G)");
    expect_contains(interp, "help", "stats_wilcoxon_signed_rank(x,y)");

    // Shapiro-Wilk: nearly-linear sample is non-normal-ish; still returns 1x2 [W, p].
    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    ASSERT_GT(interp.state().matrices.count("sw"), 0u);
    const auto& sw = interp.state().matrices.at("sw");
    ASSERT_EQ(sw.rows(), 1u);
    ASSERT_EQ(sw.cols(), 2u);
    EXPECT_GT(sw(0, 0), 0.0);
    EXPECT_LE(sw(0, 0), 1.0);
    EXPECT_GE(sw(0, 1), 0.0);
    EXPECT_LE(sw(0, 1), 1.0);

    // Mann-Whitney U: fully separated samples -> U=0, small p; 1x3 [U, z, p].
    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    ASSERT_GT(interp.state().matrices.count("mw"), 0u);
    const auto& mw = interp.state().matrices.at("mw");
    ASSERT_EQ(mw.rows(), 1u);
    ASSERT_EQ(mw.cols(), 3u);
    EXPECT_NEAR(mw(0, 0), 0.0, 1e-12);
    EXPECT_TRUE(std::isfinite(mw(0, 1)));
    EXPECT_LT(mw(0, 2), 0.05);

    // One-way ANOVA: each ROW is a group; 1x4 [F, p, df_b, df_w].
    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    ASSERT_GT(interp.state().matrices.count("an"), 0u);
    const auto& an = interp.state().matrices.at("an");
    ASSERT_EQ(an.rows(), 1u);
    ASSERT_EQ(an.cols(), 4u);
    EXPECT_GT(an(0, 0), 0.0);
    EXPECT_LT(an(0, 1), 0.05);
    EXPECT_NEAR(an(0, 2), 2.0, 1e-12);
    EXPECT_NEAR(an(0, 3), 6.0, 1e-12);

    // Wilcoxon signed-rank: clear paired shift; 1x4 [W, z, p, n_eff].
    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
    const auto& ws = interp.state().matrices.at("ws");
    ASSERT_EQ(ws.rows(), 1u);
    ASSERT_EQ(ws.cols(), 4u);
    EXPECT_TRUE(std::isfinite(ws(0, 0)));
    EXPECT_TRUE(std::isfinite(ws(0, 1)));
    EXPECT_LT(ws(0, 2), 0.05);
    EXPECT_NEAR(ws(0, 3), 8.0, 1e-12);
}

TEST(ReplCommandsTest, stats_infer_ext) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_friedman(data)");
    expect_contains(interp, "help", "stats_ks_2sample(a,b)");
    expect_contains(interp, "help", "stats_jarque_bera(x)");
    expect_contains(interp, "help", "stats_ljung_box(x,max_lag)");

    // Friedman: hand-computed 4 blocks Ã— 3 treatments â†’ chi2=1.5, df=2, p=exp(-0.75).
    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    ASSERT_GT(interp.state().matrices.count("fr"), 0u);
    const auto& fr = interp.state().matrices.at("fr");
    ASSERT_EQ(fr.rows(), 1u);
    ASSERT_EQ(fr.cols(), 3u);
    EXPECT_NEAR(fr(0, 0), 1.5, 1e-9);
    EXPECT_NEAR(fr(0, 1), 2.0, 1e-12);
    EXPECT_NEAR(fr(0, 2), std::exp(-0.75), 1e-9);

    // Two-sample KS: fully separated samples â†’ large D, small p; 1x2 [D, p].
    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    ASSERT_GT(interp.state().matrices.count("ks"), 0u);
    const auto& ks = interp.state().matrices.at("ks");
    ASSERT_EQ(ks.rows(), 1u);
    ASSERT_EQ(ks.cols(), 2u);
    EXPECT_GT(ks(0, 0), 0.9);
    EXPECT_LT(ks(0, 1), 0.05);

    // Jarque-Bera: heavily skewed exponential-like sample; 1x3 [JB, df, p].
    expect_ok(interp,
              "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 12; 14; 16; 18; 20; 25; 30; 40; 50; "
              "60; 80; 100; 150; 200; 300; 400; 500; 700; 900; 1200]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
    const auto& jb = interp.state().matrices.at("jb");
    ASSERT_EQ(jb.rows(), 1u);
    ASSERT_EQ(jb.cols(), 3u);
    EXPECT_GT(jb(0, 0), 5.0);
    EXPECT_NEAR(jb(0, 1), 2.0, 1e-12);
    EXPECT_LT(jb(0, 2), 0.05);

    // Ljung-Box: cumulative-sum random walk (same as StatsExtTest.ljung_box_cumulative_sum);
    // returns 1x3 [Q, df, p]. Pattern repeats every 3: -0.5, -0.5, 0 (80 samples).
    {
        std::string lbx = "lbx = [";
        double sum = 0.0;
        for (int i = 0; i < 80; ++i) {
            sum += ((i % 3) - 1) * 0.5;
            if (i > 0) {
                lbx += "; ";
            }
            lbx += std::to_string(sum);
        }
        lbx += "]";
        expect_ok(interp, lbx);
    }
    expect_ok(interp, "lb = stats_ljung_box(lbx, 8)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    const auto& lb = interp.state().matrices.at("lb");
    ASSERT_EQ(lb.rows(), 1u);
    ASSERT_EQ(lb.cols(), 3u);
    EXPECT_GT(lb(0, 0), 30.0);
    EXPECT_NEAR(lb(0, 1), 8.0, 1e-12);
    EXPECT_LT(lb(0, 2), 0.01);
}

TEST(ReplCommandsTest, stats_ext) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_partial_correlation(x,y,z)");
    expect_contains(interp, "help", "stats_weighted_mean(x,w)");
    expect_contains(interp, "help", "stats_trimmed_mean(x,frac)");
    expect_contains(interp, "help", "stats_arfit(x,p)");
    expect_contains(interp, "help", "stats_multiple_regression(X,y)");

    // Partial correlation with z uncorrelated to x,y -> ~ Pearson corr(x,y) = 1.
    expect_ok(interp, "px = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "py = [2; 4; 6; 8; 10; 12; 14; 16]");
    expect_ok(interp, "pz = [1; -1; 1; -1; 1; -1; 1; -1]");
    expect_ok(interp, "pc = stats_partial_correlation(px, py, pz)");
    ASSERT_GT(interp.state().scalars.count("pc"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("pc"), 1.0, 0.05);

    // Weighted mean: weights [1,1,1,1,1] on 1..5 => 3.
    expect_ok(interp, "wm = stats_weighted_mean([1; 2; 3; 4; 5], [1; 1; 1; 1; 1])");
    ASSERT_GT(interp.state().scalars.count("wm"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("wm"), 3.0, 1e-12);

    // Trimmed mean: [1..10], frac=0.1 drops one each end -> mean of 2..9 = 5.5.
    expect_ok(interp, "tm = stats_trimmed_mean([1; 2; 3; 4; 5; 6; 7; 8; 9; 10], 0.1)");
    ASSERT_GT(interp.state().scalars.count("tm"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("tm"), 5.5, 1e-12);

    // AR(1) on short series: px1 coefficient column, finite.
    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    ASSERT_GT(interp.state().matrices.count("phi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("phi").cols(), 1u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("phi")(0, 0)));
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("phi")(1, 0)));

    // Multiple regression: y = 1 + 2*x1 + 3*x2; X includes intercept column.
    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    ASSERT_GT(interp.state().matrices.count("beta"), 0u);
    EXPECT_EQ(interp.state().matrices.at("beta").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("beta").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
    EXPECT_NEAR(interp.state().matrices.at("beta")(1, 0), 2.0, 0.01);
    EXPECT_NEAR(interp.state().matrices.at("beta")(2, 0), 3.0, 0.01);
}

TEST(ReplCommandsTest, stats_weighted_ext) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_weighted_variance(x,w)");
    expect_contains(interp, "help", "stats_weighted_correlation(x,y,w)");
    expect_contains(interp, "help", "stats_bootstrap_mean(x[,n_boot[,seed]])");

    // Uniform weights: weighted variance matches sample variance.
    expect_ok(interp, "sx = [2; 4; 6; 8; 10]");
    expect_ok(interp, "sw = [1; 1; 1; 1; 1]");
    expect_ok(interp, "wv = stats_weighted_variance(sx, sw)");
    expect_ok(interp, "v = stats_var(sx)");
    ASSERT_GT(interp.state().scalars.count("wv"), 0u);
    ASSERT_GT(interp.state().scalars.count("v"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("wv"), interp.state().scalars.at("v"), 1e-12);

    // Uniform weights: weighted correlation matches Pearson correlation.
    expect_ok(interp, "cx = [1; 2; 3; 4; 5]");
    expect_ok(interp, "cy = [2; 4; 6; 8; 10]");
    expect_ok(interp, "wc = stats_weighted_correlation(cx, cy, sw)");
    expect_ok(interp, "c = stats_correlation(cx, cy)");
    ASSERT_GT(interp.state().scalars.count("wc"), 0u);
    ASSERT_GT(interp.state().scalars.count("c"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("wc"), interp.state().scalars.at("c"), 1e-12);

    // Fixed-seed bootstrap mean is deterministic.
    expect_ok(interp, "bx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "bm1 = stats_bootstrap_mean(bx, 500, 42)");
    expect_ok(interp, "bm2 = stats_bootstrap_mean(bx, 500, 42)");
    ASSERT_GT(interp.state().scalars.count("bm1"), 0u);
    ASSERT_GT(interp.state().scalars.count("bm2"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("bm1"), interp.state().scalars.at("bm2"), 1e-15);
}

TEST(ReplCommandsTest, stats_vif) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_vif(X,j)");
    expect_contains(interp, "help", "stats_variance_inflation_factor");

    // Orthogonal columns (constant vs Â±1): auxiliary R^2 ~ 0 => VIF near 1.
    expect_ok(interp, "Xo = [1, 1; 1, -1; 1, 1; 1, -1; 1, 1; 1, -1]");
    expect_ok(interp, "v0 = stats_vif(Xo, 0)");
    ASSERT_GT(interp.state().scalars.count("v0"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("v0"), 1.0, 0.2);

    expect_ok(interp, "v1 = stats_variance_inflation_factor(Xo, 1)");
    ASSERT_GT(interp.state().scalars.count("v1"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("v1"), 1.0, 0.2);
    EXPECT_NEAR(interp.state().scalars.at("v0"), interp.state().scalars.at("v1"), 1e-12);

    // Near-collinear third column -> large VIF.
    expect_ok(interp,
              "Xc = [1, 2, 3.00000001; 2, 3, 5.00000002; 3, 4, 7.00000003; "
              "4, 5, 9.00000004; 5, 6, 11.00000005]");
    expect_ok(interp, "vc = stats_vif(Xc, 2)");
    ASSERT_GT(interp.state().scalars.count("vc"), 0u);
    EXPECT_GT(interp.state().scalars.at("vc"), 100.0);
}

TEST(ReplCommandsTest, stats_variance) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_levene(G)");
    expect_contains(interp, "help", "stats_bartlett(G)");
    expect_contains(interp, "help", "stats_fligner(G)");

    // Unequal-variance groups (row = group); Levene 1x4 [F, p, df_b, df_w].
    expect_ok(interp, "G = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(G)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    const auto& lv = interp.state().matrices.at("lv");
    ASSERT_EQ(lv.rows(), 1u);
    ASSERT_EQ(lv.cols(), 4u);
    EXPECT_GT(lv(0, 0), 0.0);
    EXPECT_LT(lv(0, 1), 0.05);
    EXPECT_NEAR(lv(0, 2), 2.0, 1e-12);
    EXPECT_NEAR(lv(0, 3), 9.0, 1e-12);

    // Bartlett 1x3 [chi2, df, p].
    expect_ok(interp, "bt = stats_bartlett(G)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    const auto& bt = interp.state().matrices.at("bt");
    ASSERT_EQ(bt.rows(), 1u);
    ASSERT_EQ(bt.cols(), 3u);
    EXPECT_GT(bt(0, 0), 0.0);
    EXPECT_NEAR(bt(0, 1), 2.0, 1e-12);
    EXPECT_LT(bt(0, 2), 0.05);

    // Fligner-Killeen 1x3 [chi2, df, p].
    expect_ok(interp, "fk = stats_fligner(G)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);
    const auto& fk = interp.state().matrices.at("fk");
    ASSERT_EQ(fk.rows(), 1u);
    ASSERT_EQ(fk.cols(), 3u);
    EXPECT_GT(fk(0, 0), 0.0);
    EXPECT_NEAR(fk(0, 1), 2.0, 1e-12);
    EXPECT_LT(fk(0, 2), 0.05);
}

TEST(ReplCommandsTest, stats_ks_norm) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_ks_norm(x,mu,sigma)");

    expect_ok(interp, "ksn = stats_ks_norm([0; 1; 2; 3; 4; 5; 6; 7], 0, 1)");
    EXPECT_GT(interp.state().scalars.at("ksn"), 0.3);
    EXPECT_LT(interp.state().scalars.at("ksn"), 1.0);

    expect_ok(interp, "stats_ks_norm([0; 1; 2; 3; 4; 5; 6; 7], 0, 1)");
}

TEST(ReplCommandsTest, stats_kde_bootstrap) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_kde(samples,grid,h[,kernel])");
    expect_contains(interp, "help", "stats_bootstrap_ci(x)");
    expect_contains(interp, "help", "[point, lower, upper, std_error]");

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    ASSERT_GT(interp.state().matrices.count("k"), 0u);
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("k").cols(), 1u);

    expect_ok(interp,
              "ke = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1, \"epanechnikov\")");
    ASSERT_GT(interp.state().matrices.count("ke"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ke").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("ke").cols(), 1u);
    for (size_t i = 0; i < 7; ++i) {
        EXPECT_TRUE(std::isfinite(interp.state().matrices.at("ke")(i, 0)));
    }

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    ASSERT_GT(interp.state().matrices.count("ci"), 0u);
    const auto& ci = interp.state().matrices.at("ci");
    EXPECT_EQ(ci.rows(), 1u);
    EXPECT_EQ(ci.cols(), 4u);
    EXPECT_NEAR(ci(0, 0), 5.5, 1e-9);
    EXPECT_LT(ci(0, 1), ci(0, 0));
    EXPECT_GT(ci(0, 2), ci(0, 0));
    EXPECT_GT(ci(0, 3), 0.0);
}

TEST(ReplCommandsTest, stats_geo_image) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").rows(), 1u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "L = label_components(M)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);
}

TEST(ReplCommandsTest, image_stats) {
    Interpreter interp;

    // Bright quadrant corner -> Harris keypoints (tail11 routing smoke).
    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    ASSERT_GT(interp.state().matrices.at("H").rows(), 0u);

    expect_ok(interp, "k = stats_kde([0; 1; 2], [-1; 0; 1; 2], 1)");
    EXPECT_GT(interp.state().matrices.at("k").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "RGB = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB").cols(), 3u);
}

TEST(ReplCommandsTest, kruskal_shapiro) {
    Interpreter interp;

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);
}

TEST(ReplCommandsTest, mannwhitney_ks) {
    Interpreter interp;

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);
}

TEST(ReplCommandsTest, anova_variance) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);
}

TEST(ReplCommandsTest, wilcoxon_friedman_jarque_ljung) {
    Interpreter interp;

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, hull3d_linreg) {
    Interpreter interp;

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, hull3d_linreg_2) {
    Interpreter interp;

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_2) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_2) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_2) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_2) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_2) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_2) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_2) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_2) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_2) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_2) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_3) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_3) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_3) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_3) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_3) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_3) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_3) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_3) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_3) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_3) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_2) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_4) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_4) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_4) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_4) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_4) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_4) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_4) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_4) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_4) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_4) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_3) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_5) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_5) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_5) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_5) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_5) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_5) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_5) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_5) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_5) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_5) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_4) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_6) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_6) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_6) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_6) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_6) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_6) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_6) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_6) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_6) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_6) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_5) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_7) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_7) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_7) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_7) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_7) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_7) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_7) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_7) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_7) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_7) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_6) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_8) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_8) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_8) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_8) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_8) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_8) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_8) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_8) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_8) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_8) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_7) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_9) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_9) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_9) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_9) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_9) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_9) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_9) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_9) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_9) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_9) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_8) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_10) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_10) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_10) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_10) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_10) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_10) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_10) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_10) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_10) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_10) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_9) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_11) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_11) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_11) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_11) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_11) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_11) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_11) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_11) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_11) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_11) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_10) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_12) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_12) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_12) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_12) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_12) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_12) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_12) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_12) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_12) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_12) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_11) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_13) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_13) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_13) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_13) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_13) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_13) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_13) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_13) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_13) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_13) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_12) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_14) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_14) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_14) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_14) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_14) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_14) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_14) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_14) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_14) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_14) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_13) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_15) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_15) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_15) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_15) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_15) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_15) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_15) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_15) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_15) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_15) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_14) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_16) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_16) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_16) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_16) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_16) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_16) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_16) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_16) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_16) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_16) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_15) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_17) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_17) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_17) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_17) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_17) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_17) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_17) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_17) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_17) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_17) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_16) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_18) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_18) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_18) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_18) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_18) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_18) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_18) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_18) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_18) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_18) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_17) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_19) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_19) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_19) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_19) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_19) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_19) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_19) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_19) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_19) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_19) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_18) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_20) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_20) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_20) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_20) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_20) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_20) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_20) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_20) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_20) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_20) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_19) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_21) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_21) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_21) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_21) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_21) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_21) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_21) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_21) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_21) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_21) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_20) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_22) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_22) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_22) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_22) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_22) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_22) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_22) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_22) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_22) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_22) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_21) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_23) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_23) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_23) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_23) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_23) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_23) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_23) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_23) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_23) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_23) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_22) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_24) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_24) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_24) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_24) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_24) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_24) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_24) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_24) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_24) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_24) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_23) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_25) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_25) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_25) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_25) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_25) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_25) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_25) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_25) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_25) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_25) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_24) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_26) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_26) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_26) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_26) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_26) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_26) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_26) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_26) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_26) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_26) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_25) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_27) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_27) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_27) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_27) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_27) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_27) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_27) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_27) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_27) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_27) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_26) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_28) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_28) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_28) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_28) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_28) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_28) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_28) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_28) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_28) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_28) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_27) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_29) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_29) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_29) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_29) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_29) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_29) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_29) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_29) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_29) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_29) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_28) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_30) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_30) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_30) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_30) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_30) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_30) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_30) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_30) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_30) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_30) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_29) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_31) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_31) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_31) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_31) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_31) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_31) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_31) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_31) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_31) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_31) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_30) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_32) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_32) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_32) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, shapiro_mannwhitney_32) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(ReplCommandsTest, ks_delaunay_32) {
    Interpreter interp;

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(ReplCommandsTest, label_anova_32) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(ReplCommandsTest, levene_bartlett_32) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(ReplCommandsTest, fligner_wilcoxon_32) {
    Interpreter interp;

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(ReplCommandsTest, friedman_jarque_32) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(ReplCommandsTest, ljung_box_32) {
    Interpreter interp;

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(ReplCommandsTest, linreg_31) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, pacf_slic_33) {
    Interpreter interp;

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(ReplCommandsTest, kde_bootstrap_33) {
    Interpreter interp;

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(ReplCommandsTest, arfit_multireg_33) {
    Interpreter interp;

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(ReplCommandsTest, stats_bootstrap_mean_errors) {
    Interpreter interp;

    expect_ok(interp, "stats_bootstrap_mean([1; 2; 3; 4; 5], 20, 1)");

    expect_error_contains(interp, "stats_bootstrap_mean(missing, 20, 1)", "unknown matrix");
    expect_error_contains(interp, "stats_bootstrap_mean([1; 2; 3; 4; 5], 0, 1)",
                          "expected positive integer n_boot");
    expect_error_contains(interp, "stats_bootstrap_mean([1; 2; 3; 4; 5], 1.5, 1)",
                          "expected positive integer n_boot");
    expect_error_contains(interp, "stats_bootstrap_mean([1; 2; 3; 4; 5], notnum, 1)",
                          "expected stats_bootstrap_mean(x[, n_boot[, seed]])");
    expect_error_contains(interp, "stats_bootstrap_mean([1; 2; 3; 4; 5], 20, -1)",
                          "expected non-negative integer seed");
    expect_error_contains(interp, "bm = stats_bootstrap_mean()",
                          "expected stats_bootstrap_mean(x[, n_boot[, seed]])");
    expect_error_contains(interp, "bm = stats_bootstrap_mean(missing)", "unknown matrix");
    expect_error_contains(interp, "bm = stats_bootstrap_mean([1; 2; 3; 4; 5], 0)",
                          "expected positive integer n_boot");
    expect_error_contains(interp, "bm = stats_bootstrap_mean([1; 2; 3; 4; 5], 20, -1)",
                          "expected non-negative integer seed");
}

TEST(ReplCommandsTest, kruskal_wallis_noassign) {
    Interpreter interp;
    expect_contains(interp, "kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])",
                    "kruskal_wallis");
    expect_error_contains(interp, "kruskal_wallis([1, 2, 3])", "at least two group rows");
}

TEST(ReplCommandsTest, stats_mean_noassign) {
    Interpreter interp;
    expect_contains(interp, "stats_mean([1; 2; 3; 4; 5])", "3");
    expect_error_contains(interp, "stats_mean(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, stats_median_noassign) {
    Interpreter interp;
    expect_contains(interp, "stats_median([1; 2; 3; 4; 5])", "3");
    expect_error_contains(interp, "stats_median(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, stats_var_noassign) {
    Interpreter interp;
    expect_contains(interp, "stats_var([1; 2; 3; 4; 5])", "2.5");
    expect_error_contains(interp, "stats_var(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, stats_stddev_noassign) {
    Interpreter interp;
    expect_contains(interp, "stats_stddev([1; 2; 3; 4; 5])", "1.58114");
    expect_error_contains(interp, "stats_stddev(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, stats_min_value_noassign) {
    Interpreter interp;
    expect_contains(interp, "stats_min_value([1; 2; 3; 4; 5])", "1");
    expect_error_contains(interp, "stats_min_value(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, stats_max_value_noassign) {
    Interpreter interp;
    expect_contains(interp, "stats_max_value([1; 2; 3; 4; 5])", "5");
    expect_error_contains(interp, "stats_max_value(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, stats_rms_noassign) {
    Interpreter interp;
    expect_ok(interp, "stats_rms([3; 4])");
    expect_error_contains(interp, "stats_rms(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, stats_skewness_noassign) {
    Interpreter interp;
    expect_contains(interp, "stats_skewness([1; 2; 3; 4; 5])", "0");
    expect_error_contains(interp, "stats_skewness(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, stats_kurtosis_noassign) {
    Interpreter interp;
    expect_ok(interp, "stats_kurtosis([1; 2; 3; 4; 5])");
    expect_error_contains(interp, "stats_kurtosis(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, stats_mode_noassign) {
    Interpreter interp;
    expect_contains(interp, "stats_mode([1; 2; 2; 3])", "2");
    expect_error_contains(interp, "stats_mode(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, stats_geometric_mean_noassign) {
    Interpreter interp;
    expect_contains(interp, "stats_geometric_mean([2; 8])", "4");
    expect_error_contains(interp, "stats_geometric_mean(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, stats_harmonic_mean_noassign) {
    Interpreter interp;
    expect_contains(interp, "stats_harmonic_mean([1; 2; 3; 4])", "1.92");
    expect_error_contains(interp, "stats_harmonic_mean(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, stats_mad_noassign) {
    Interpreter interp;
    expect_contains(interp, "stats_mad([1; 1; 2; 2; 4; 6; 9])", "1.4826");
    expect_error_contains(interp, "stats_mad(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, stats_iqr_noassign) {
    Interpreter interp;
    expect_contains(interp, "stats_iqr([1; 2; 3; 4; 5; 6; 7; 8; 9])", "4");
    expect_error_contains(interp, "stats_iqr(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, stats_ttest_noassign) {
    Interpreter interp;
    expect_ok(interp, "stats_ttest([6; 7; 8; 9; 10], 5)");
    expect_error_contains(interp, "stats_ttest(no_such_matrix, 5)", "unknown matrix");
}

TEST(ReplCommandsTest, stats_trimmed_mean_noassign) {
    Interpreter interp;
    expect_ok(interp, "stats_trimmed_mean([1; 2; 3; 4; 5; 6; 7; 8; 9; 10], 0.1)");
    expect_error_contains(interp, "stats_trimmed_mean(no_such_matrix, 0.1)", "unknown matrix");
}

TEST(ReplCommandsTest, stats_vif_noassign) {
    Interpreter interp;
    expect_ok(interp, "stats_vif([1, 1; 1, -1; 1, 1; 1, -1; 1, 1; 1, -1], 0)");
    expect_error_contains(interp, "stats_vif(no_such_matrix, 0)", "unknown matrix");
}
