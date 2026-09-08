#define _USE_MATH_DEFINES
#include "ms/ml/ml.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

using namespace ms::ml;

// ---- Shared deterministic fixtures (no RNG, so identical on every platform) ----

// Three tight, well-separated blobs: 10 points each, 5 features, blob radius
// ~0.35, blob-centre separation 40. The jitter is trigonometric rather than
// drawn from a distribution, so it does not depend on any standard library.
static Mat tsne_three_blobs() {
    const double cx[3] = {0.0, 40.0, 0.0};
    const double cy[3] = {0.0, 0.0, 40.0};
    Mat X;
    X.reserve(30);
    for (int c = 0; c < 3; ++c) {
        for (int t = 0; t < 10; ++t) {
            const double s = static_cast<double>(c * 10 + t);
            const double a = 0.35 * std::sin(1.7 * s + 0.3 * static_cast<double>(c));
            const double b = 0.35 * std::cos(2.3 * s + 0.7 * static_cast<double>(c));
            X.push_back({cx[c] + a, cy[c] + b, a - b, 0.5 * b, 0.25 * a});
        }
    }
    return X;
}

// Mean within-blob and mean between-blob pairwise distance in the embedding,
// for `nb` blobs of `per` points laid out contiguously.
static void tsne_blob_stats(const Mat& Y, int nb, int per, double& within,
                            double& between) {
    double sw = 0.0, sb = 0.0;
    int cw = 0, cb = 0;
    const int n = nb * per;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double d = 0.0;
            for (size_t a = 0; a < Y[static_cast<size_t>(i)].size(); ++a) {
                const double t = Y[static_cast<size_t>(i)][a] - Y[static_cast<size_t>(j)][a];
                d += t * t;
            }
            d = std::sqrt(d);
            if (i / per == j / per) { sw += d; ++cw; } else { sb += d; ++cb; }
        }
    }
    within = (cw > 0) ? sw / static_cast<double>(cw) : 0.0;
    between = (cb > 0) ? sb / static_cast<double>(cb) : 0.0;
}

static bool tsne_all_finite(const Mat& Y) {
    for (const auto& r : Y)
        for (double v : r)
            if (!std::isfinite(v)) return false;
    return true;
}

// Every point of tsne_three_blobs() is nearer its own blob's centroid than any
// other blob's — a stricter statement than a within/between ratio.
static bool tsne_centroids_separate(const Mat& Y) {
    Mat cen(3, Vec(2, 0.0));
    for (int c = 0; c < 3; ++c)
        for (int t = 0; t < 10; ++t)
            for (int a = 0; a < 2; ++a)
                cen[static_cast<size_t>(c)][static_cast<size_t>(a)] +=
                    Y[static_cast<size_t>(c * 10 + t)][static_cast<size_t>(a)] / 10.0;
    for (int i = 0; i < 30; ++i) {
        int best = 0;
        double bd = 1e300;
        for (int c = 0; c < 3; ++c) {
            double d = 0.0;
            for (int a = 0; a < 2; ++a) {
                const double u = Y[static_cast<size_t>(i)][static_cast<size_t>(a)] -
                                 cen[static_cast<size_t>(c)][static_cast<size_t>(a)];
                d += u * u;
            }
            if (d < bd) { bd = d; best = c; }
        }
        if (best != i / 10) return false;
    }
    return true;
}

// ---- Degenerate shapes ----

TEST(MLTSNEBarnesHut, EmptyInput) {
    TSNE t(2, 30.0, 200.0, 100, 42);
    Mat Y = t.fit_transform(Mat{});
    EXPECT_TRUE(Y.empty());
}

TEST(MLTSNEBarnesHut, SinglePointIsZeroRow) {
    Mat X = {{1.0, 2.0, 3.0}};
    TSNE t(2, 30.0, 200.0, 500, 42);
    Mat Y = t.fit_transform(X);
    ASSERT_EQ(Y.size(), 1u);
    ASSERT_EQ(Y[0].size(), 2u);
    EXPECT_DOUBLE_EQ(Y[0][0], 0.0);
    EXPECT_DOUBLE_EQ(Y[0][1], 0.0);
}

TEST(MLTSNEBarnesHut, ZeroComponentsDegradesToEmptyRows) {
    Mat X = {{0.0, 0.0}, {1.0, 1.0}, {2.0, 0.0}};
    TSNE t(0, 2.0, 100.0, 10, 1);
    const Mat Y = t.fit_transform(X);
    ASSERT_EQ(Y.size(), 3u);
    for (const auto& r : Y) EXPECT_TRUE(r.empty());
}

// n == 2 also exercises k == 1, an entropy that is identically 0 for every
// beta, and the halving branch of the bisection all at once.
TEST(MLTSNEBarnesHut, TwoPointsSeparateAndCentre) {
    Mat X = {{0.0, 0.0}, {5.0, 0.0}};
    TSNE t(2, 5.0, 200.0, 300, 7);  // perplexity 5 >= n: unreachable target
    const Mat Y = t.fit_transform(X);
    ASSERT_EQ(Y.size(), 2u);
    ASSERT_EQ(Y[0].size(), 2u);
    EXPECT_TRUE(tsne_all_finite(Y));
    EXPECT_NEAR(Y[0][0] + Y[1][0], 0.0, 1e-9);
    EXPECT_NEAR(Y[0][1] + Y[1][1], 0.0, 1e-9);
    double d = 0.0;
    for (int a = 0; a < 2; ++a) {
        const double u = Y[0][static_cast<size_t>(a)] - Y[1][static_cast<size_t>(a)];
        d += u * u;
    }
    EXPECT_GT(std::sqrt(d), 1e-3);
}

TEST(MLTSNEBarnesHut, PerplexityLargerThanSampleCount) {
    Mat X = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {5.0, 5.0}};
    TSNE t(2, 50.0, 100.0, 100, 42);  // k clamps to n-1 = 4, log(50) unreachable
    const Mat Y = t.fit_transform(X);
    ASSERT_EQ(Y.size(), 5u);
    ASSERT_EQ(Y[0].size(), 2u);
    EXPECT_TRUE(tsne_all_finite(Y));
}

TEST(MLTSNEBarnesHut, HighComponentCountFallsBackToExact) {
    Mat X = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    TSNE t(4, 2.0, 100.0, 50, 3);  // n_components > n and > the tree's dim cap
    const Mat Y = t.fit_transform(X);
    ASSERT_EQ(Y.size(), 3u);
    for (const auto& r : Y) EXPECT_EQ(r.size(), 4u);
    EXPECT_TRUE(tsne_all_finite(Y));
}

TEST(MLTSNEBarnesHut, ZeroIterationsReturnsCentredInit) {
    const Mat X = tsne_three_blobs();
    TSNE t(2, 5.0, 200.0, 0, 77);
    Vec trace;
    const Mat Y = t.fit_transform(X, trace);
    ASSERT_EQ(Y.size(), 30u);
    ASSERT_EQ(Y[0].size(), 2u);
    EXPECT_EQ(trace.size(), 1u);  // only the final sample
    for (int a = 0; a < 2; ++a) {
        double s = 0.0;
        for (const auto& r : Y) s += r[static_cast<size_t>(a)];
        EXPECT_NEAR(s, 0.0, 1e-12);
    }
    // The initialisation is N(0, 1e-4); 1e-2 is a 100-sigma bound.
    for (const auto& r : Y)
        for (double v : r) EXPECT_LT(std::abs(v), 1e-2);
}

// ---- Behaviour ----

TEST(MLTSNEBarnesHut, BlobsStaySeparated) {
    const Mat X = tsne_three_blobs();
    TSNE t(2, 5.0, 200.0, 1000, 7);
    const Mat Y = t.fit_transform(X);
    ASSERT_EQ(Y.size(), 30u);
    ASSERT_EQ(Y[0].size(), 2u);
    ASSERT_TRUE(tsne_all_finite(Y));
    double within = 0.0, between = 0.0;
    tsne_blob_stats(Y, 3, 10, within, between);
    EXPECT_GT(between, within);
    EXPECT_GT(between, 1.5 * within);
    EXPECT_TRUE(tsne_centroids_separate(Y));
}

TEST(MLTSNEBarnesHut, KLDivergenceDecreases) {
    const Mat X = tsne_three_blobs();
    TSNE t(2, 5.0, 200.0, 1000, 7);
    Vec trace;
    const Mat Y = t.fit_transform(X, trace);
    ASSERT_EQ(trace.size(), 21u);  // ceil(1000/50) + 1
    for (double v : trace) EXPECT_TRUE(std::isfinite(v));
    for (double v : trace) EXPECT_GE(v, 0.0);  // KL is non-negative
    // trace[1] is the objective at iteration 50, still inside early
    // exaggeration; the last entry belongs to the returned embedding.
    EXPECT_LT(trace.back(), trace[1]);
    EXPECT_LT(trace.back(), trace.front());
    // The trace has settled by the end of the run.
    EXPECT_LT(trace.back(), 0.5);
    // The accessor must agree with the final trace entry.
    EXPECT_NEAR(t.kl_divergence(X, Y), trace.back(), 1e-9);
}

TEST(MLTSNEBarnesHut, KLDivergenceRanksEmbeddings) {
    const Mat X = tsne_three_blobs();
    TSNE t(2, 5.0, 200.0, 1000, 7);
    const Mat Y = t.fit_transform(X);
    const double fitted = t.kl_divergence(X, Y);
    EXPECT_GE(fitted, 0.0);
    // A collapsed embedding makes Q uniform and must score far worse.
    const Mat collapsed(30, Vec(2, 0.0));
    EXPECT_GT(t.kl_divergence(X, collapsed), fitted);
    EXPECT_GT(t.kl_divergence(X, collapsed), 10.0 * fitted);
    // Documented guards: all of these return exactly 0.
    EXPECT_DOUBLE_EQ(t.kl_divergence(X, Mat{{0.0, 0.0}}), 0.0);          // size mismatch
    EXPECT_DOUBLE_EQ(t.kl_divergence(X, Mat(30, Vec(1, 0.0))), 0.0);     // rows too narrow
    EXPECT_DOUBLE_EQ(t.kl_divergence(Mat{{1.0, 2.0}}, Mat{{0.0, 0.0}}), 0.0);  // n < 2
}

TEST(MLTSNEBarnesHut, DeterministicForFixedSeed) {
    const Mat X = tsne_three_blobs();
    TSNE a(2, 5.0, 200.0, 120, 1234);
    TSNE b(2, 5.0, 200.0, 120, 1234);
    const Mat Ya = a.fit_transform(X);
    const Mat Yb = b.fit_transform(X);
    ASSERT_EQ(Ya.size(), Yb.size());
    for (size_t i = 0; i < Ya.size(); ++i) {
        ASSERT_EQ(Ya[i].size(), Yb[i].size());
        for (size_t j = 0; j < Ya[i].size(); ++j)
            EXPECT_EQ(Ya[i][j], Yb[i][j]);  // bitwise, not EXPECT_NEAR
    }
    TSNE c(2, 5.0, 200.0, 120, 4321);
    const Mat Yc = c.fit_transform(X);
    bool any_diff = false;
    for (size_t i = 0; i < Ya.size() && !any_diff; ++i)
        for (size_t j = 0; j < Ya[i].size(); ++j)
            if (Ya[i][j] != Yc[i][j]) { any_diff = true; break; }
    EXPECT_TRUE(any_diff);
}

TEST(MLTSNEBarnesHut, ThetaZeroAndHalfAgreeOnStructure) {
    const Mat X = tsne_three_blobs();
    TSNE exact(2, 5.0, 200.0, 1000, 11);
    exact.theta = 0.0;  // exact O(n^2) repulsion
    TSNE bh(2, 5.0, 200.0, 1000, 11);
    bh.theta = 0.5;  // Barnes-Hut
    const Mat Ye = exact.fit_transform(X);
    const Mat Yb = bh.fit_transform(X);
    ASSERT_TRUE(tsne_all_finite(Ye));
    ASSERT_TRUE(tsne_all_finite(Yb));
    double we = 0.0, be = 0.0, wb = 0.0, bb = 0.0;
    tsne_blob_stats(Ye, 3, 10, we, be);
    tsne_blob_stats(Yb, 3, 10, wb, bb);
    EXPECT_GT(be, 1.5 * we);
    EXPECT_GT(bb, 1.5 * wb);
    // The two embeddings are never compared numerically: the approximation and
    // the summation order both differ. Only the cluster structure must agree.
    EXPECT_TRUE(tsne_centroids_separate(Ye));
    EXPECT_TRUE(tsne_centroids_separate(Yb));
}

// theta is clamped to [0, 1], and the clamp is exact: an out-of-range value
// must reproduce the boundary run bit for bit.
TEST(MLTSNEBarnesHut, ThetaIsClampedToUnitInterval) {
    const Mat X = tsne_three_blobs();
    TSNE zero(2, 5.0, 200.0, 200, 3);
    zero.theta = 0.0;
    TSNE negative(2, 5.0, 200.0, 200, 3);
    negative.theta = -1.0;
    TSNE nan_theta(2, 5.0, 200.0, 200, 3);
    nan_theta.theta = std::nan("");
    TSNE one(2, 5.0, 200.0, 200, 3);
    one.theta = 1.0;
    TSNE over(2, 5.0, 200.0, 200, 3);
    over.theta = 5.0;

    const Mat Yzero = zero.fit_transform(X);
    const Mat Yneg = negative.fit_transform(X);
    const Mat Ynan = nan_theta.fit_transform(X);
    const Mat Yone = one.fit_transform(X);
    const Mat Yover = over.fit_transform(X);
    ASSERT_EQ(Yzero.size(), 30u);
    for (size_t i = 0; i < Yzero.size(); ++i) {
        for (size_t j = 0; j < 2u; ++j) {
            EXPECT_EQ(Yneg[i][j], Yzero[i][j]);
            EXPECT_EQ(Ynan[i][j], Yzero[i][j]);
            EXPECT_EQ(Yover[i][j], Yone[i][j]);
        }
    }
}

TEST(MLTSNEBarnesHut, DuplicatePointsAreFinite) {
    Mat X = {{0.0, 0.0}, {0.0, 0.0},   {0.0, 0.0},
             {10.0, 10.0}, {10.0, 10.0}, {10.0, 10.0}};
    TSNE t(2, 2.0, 200.0, 1000, 5);
    const Mat Y = t.fit_transform(X);
    ASSERT_EQ(Y.size(), 6u);
    ASSERT_TRUE(tsne_all_finite(Y));
    double within = 0.0, between = 0.0;
    tsne_blob_stats(Y, 2, 3, within, between);
    EXPECT_GT(between, within);
    EXPECT_GT(between, 10.0 * within);
}

// Every embedding point identical is the case that makes the tree's root
// bounding box degenerate and forces the depth cap to bucket coincident points.
TEST(MLTSNEBarnesHut, AllPointsIdenticalStaysFinite) {
    const Mat X(20, Vec(3, 2.5));
    TSNE t(2, 5.0, 200.0, 200, 4);
    const Mat Y = t.fit_transform(X);
    ASSERT_EQ(Y.size(), 20u);
    ASSERT_EQ(Y[0].size(), 2u);
    EXPECT_TRUE(tsne_all_finite(Y));
}

TEST(MLTSNEBarnesHut, ThreeComponentsUsesOctree) {
    const Mat X = tsne_three_blobs();
    TSNE t(3, 5.0, 200.0, 1000, 9);
    const Mat Y = t.fit_transform(X);
    ASSERT_EQ(Y.size(), 30u);
    for (const auto& r : Y) ASSERT_EQ(r.size(), 3u);
    EXPECT_TRUE(tsne_all_finite(Y));
    double within = 0.0, between = 0.0;
    tsne_blob_stats(Y, 3, 10, within, between);
    EXPECT_GT(between, within);
    EXPECT_GT(between, 1.5 * within);
    for (int a = 0; a < 3; ++a) {
        double s = 0.0;
        for (const auto& r : Y) s += r[static_cast<size_t>(a)];
        EXPECT_NEAR(s, 0.0, 1e-6);
    }
}

TEST(MLTSNEBarnesHut, ExplicitNeighborCount) {
    const Mat X = tsne_three_blobs();
    TSNE autok(2, 5.0, 200.0, 1000, 21);  // k = floor(3 * 5) = 15
    TSNE fixedk(2, 5.0, 200.0, 1000, 21);
    fixedk.n_neighbors = 4;
    const Mat Ya = autok.fit_transform(X);
    const Mat Yf = fixedk.fit_transform(X);
    ASSERT_EQ(Ya.size(), Yf.size());
    EXPECT_TRUE(tsne_all_finite(Ya));
    EXPECT_TRUE(tsne_all_finite(Yf));
    double w1 = 0.0, b1 = 0.0, w2 = 0.0, b2 = 0.0;
    tsne_blob_stats(Ya, 3, 10, w1, b1);
    tsne_blob_stats(Yf, 3, 10, w2, b2);
    EXPECT_GT(b1, w1);
    EXPECT_GT(b2, w2);
    // Different neighbourhood sizes give genuinely different embeddings.
    bool differs = false;
    for (size_t i = 0; i < Ya.size() && !differs; ++i)
        for (size_t j = 0; j < Ya[i].size(); ++j)
            if (Ya[i][j] != Yf[i][j]) { differs = true; break; }
    EXPECT_TRUE(differs);
}

// ---- Defensive degradation on hostile configuration ----

TEST(MLTSNEBarnesHut, NonFiniteAndNonPositiveParametersDegrade) {
    const Mat X = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0},  {1.0, 1.0},
                   {5.0, 5.0}, {5.5, 5.0}, {0.5, 0.5}};
    const double inf = std::numeric_limits<double>::infinity();
    const double nan_v = std::nan("");
    struct Case { double perplexity, lr, exaggeration; int neighbors, iters; };
    const Case cases[] = {
        {0.0, 200.0, 12.0, 0, 100},    {-5.0, 200.0, 12.0, 0, 100},
        {inf, 200.0, 12.0, 0, 100},    {nan_v, 200.0, 12.0, 0, 100},
        {1e300, 200.0, 12.0, 0, 100},  {0.5, 200.0, 12.0, 0, 100},
        {3.0, 0.0, 12.0, 0, 100},      {3.0, -1.0, 12.0, 0, 100},
        {3.0, nan_v, 12.0, 0, 100},    {3.0, 200.0, 0.0, 0, 100},
        {3.0, 200.0, nan_v, 0, 100},   {3.0, 200.0, 12.0, -3, 100},
        {3.0, 200.0, 12.0, 1, 100},    {3.0, 200.0, 12.0, 1000000, 100},
        {3.0, 200.0, 12.0, 0, -5},
    };
    for (const Case& c : cases) {
        TSNE t(2, c.perplexity, c.lr, c.iters, 42);
        t.n_neighbors = c.neighbors;
        t.early_exaggeration = c.exaggeration;
        const Mat Y = t.fit_transform(X);
        ASSERT_EQ(Y.size(), 7u);
        ASSERT_EQ(Y[0].size(), 2u);
        EXPECT_TRUE(tsne_all_finite(Y));
    }
}

TEST(MLTSNEBarnesHut, RaggedAndEmptyRowsAreHandled) {
    // Distances use the shortest common prefix of the two rows.
    const Mat ragged = {{1.0, 2.0, 3.0}, {4.0}, {5.0, 6.0}, {7.0, 8.0, 9.0, 10.0}};
    TSNE t(2, 2.0, 200.0, 100, 1);
    const Mat Y = t.fit_transform(ragged);
    ASSERT_EQ(Y.size(), 4u);
    ASSERT_EQ(Y[0].size(), 2u);
    EXPECT_TRUE(tsne_all_finite(Y));

    // Featureless rows make every distance 0, so P is uniform over k.
    const Mat featureless(5, Vec{});
    const Mat Y2 = t.fit_transform(featureless);
    ASSERT_EQ(Y2.size(), 5u);
    ASSERT_EQ(Y2[0].size(), 2u);
    EXPECT_TRUE(tsne_all_finite(Y2));
}

TEST(MLTSNEBarnesHut, LargerDatasetStaysFinite) {
    // 120 points across 4 offset groups: enough to build a tree several levels
    // deep and to exercise the summarisation branch of the traversal.
    Mat X;
    X.reserve(120);
    for (int i = 0; i < 120; ++i) {
        const double s = static_cast<double>(i);
        const double g = static_cast<double>(i % 4) * 25.0;
        X.push_back({g + std::sin(1.3 * s), g + std::cos(0.7 * s), std::sin(2.1 * s)});
    }
    TSNE t(2, 15.0, 200.0, 300, 2024);
    const Mat Y = t.fit_transform(X);
    ASSERT_EQ(Y.size(), 120u);
    ASSERT_EQ(Y[0].size(), 2u);
    EXPECT_TRUE(tsne_all_finite(Y));
    EXPECT_GE(t.kl_divergence(X, Y), 0.0);
    for (int a = 0; a < 2; ++a) {
        double s = 0.0;
        for (const auto& r : Y) s += r[static_cast<size_t>(a)];
        EXPECT_NEAR(s, 0.0, 1e-6);
    }
}
