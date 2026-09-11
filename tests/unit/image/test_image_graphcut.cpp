// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "ms/image/image.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <utility>
#include <vector>

using namespace ms::image;

// ---- helpers ----

// Single-channel seed mask with the listed (row, col) pixels marked.
static Image gc_mask(int rows, int cols, std::initializer_list<std::pair<int, int>> pts) {
    Image m(rows, cols, 1, 0.f);
    for (const auto& p : pts) m.at(p.first, p.second, 0) = 1.f;
    return m;
}

// Compare a returned mask against one string per row ('1' = foreground).
static void gc_expect_mask(const Image& got, std::initializer_list<const char*> want) {
    ASSERT_EQ(got.rows, static_cast<int>(want.size()));
    ASSERT_EQ(got.channels, 1);
    int r = 0;
    for (const char* row : want) {
        ASSERT_EQ(static_cast<int>(std::strlen(row)), got.cols) << "row " << r;
        for (int c = 0; c < got.cols; ++c)
            EXPECT_FLOAT_EQ(got.at(r, c, 0), row[c] == '1' ? 1.f : 0.f)
                << "mismatch at (" << r << "," << c << ")";
        ++r;
    }
}

static int gc_count_fg(const Image& m) {
    int n = 0;
    for (const float v : m.data)
        if (v > 0.5f) ++n;
    return n;
}

// Independent re-derivation of the documented energy model, used to check that
// min_cut_value() really is the energy of the mask graph_cut_segment() returns
// (and that no fixed labelling beats it). Deliberately written from the header
// documentation rather than shared with the implementation.
static std::vector<double> gc_plane(const Image& im) {
    std::vector<double> out;
    if (im.empty()) return out;
    const std::size_t n = static_cast<std::size_t>(im.rows) * static_cast<std::size_t>(im.cols);
    out.resize(n);
    if (im.channels >= 3) {
        const Image g = rgb2gray(im);
        for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<double>(g.data[i]);
    } else {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = static_cast<double>(im.data[i * static_cast<std::size_t>(im.channels)]);
    }
    return out;
}

static double gc_energy_of(const Image& gray, const Image& fg, const Image& bg, double lambda,
                           double sigma, int connectivity, const Image& mask) {
    const int R = gray.rows, C = gray.cols, N = R * C;
    const std::vector<double> I = gc_plane(gray), F = gc_plane(fg), B = gc_plane(bg);

    std::vector<char> is_fg(static_cast<std::size_t>(N), 0), is_bg(static_cast<std::size_t>(N), 0);
    int n_fg = 0, n_bg = 0;
    for (int i = 0; i < N; ++i) {
        const std::size_t u = static_cast<std::size_t>(i);
        if (!F.empty() && F[u] >= 0.5) { is_fg[u] = 1; ++n_fg; }
        else if (!B.empty() && B[u] >= 0.5) { is_bg[u] = 1; ++n_bg; }
    }

    double mu_fg = 0.0, mu_bg = 0.0;
    if (n_fg > 0 || n_bg > 0) {
        double s_fg = 0.0, s_bg = 0.0, s_other = 0.0;
        int n_other = 0;
        for (int i = 0; i < N; ++i) {
            const std::size_t u = static_cast<std::size_t>(i);
            if (is_fg[u]) s_fg += I[u];
            else if (is_bg[u]) s_bg += I[u];
            else { s_other += I[u]; ++n_other; }
        }
        // A seeded mean comes from its own seeds; an unseeded one from every
        // pixel the other mask does not cover.
        if (n_fg > 0 && n_bg > 0) {
            mu_fg = s_fg / n_fg;
            mu_bg = s_bg / n_bg;
        } else if (n_fg > 0) {
            mu_fg = s_fg / n_fg;
            mu_bg = n_other > 0 ? s_other / n_other : mu_fg;
        } else {
            mu_bg = s_bg / n_bg;
            mu_fg = n_other > 0 ? s_other / n_other : mu_bg;
        }
    } else {
        double a = I[0], b = I[0];
        for (const double x : I) { a = std::min(a, x); b = std::max(b, x); }
        for (int it = 0; it < 100; ++it) {
            const double mid = 0.5 * (a + b);
            double sa = 0.0, sb = 0.0;
            int na = 0, nb = 0;
            for (const double x : I) {
                if (x <= mid) { sa += x; ++na; }
                else          { sb += x; ++nb; }
            }
            const double a2 = na > 0 ? sa / na : a;
            const double b2 = nb > 0 ? sb / nb : b;
            const double move = std::max(std::abs(a2 - a), std::abs(b2 - b));
            a = a2;
            b = b2;
            if (move < 1e-12) break;
        }
        mu_bg = std::min(a, b);
        mu_fg = std::max(a, b);
    }

    const double lam = lambda > 0.0 ? lambda : 0.0;
    const int n_off = connectivity == 8 ? 4 : 2;
    const int dr[4] = {0, 1, 1, 1};
    const int dc[4] = {1, 0, 1, -1};
    const double inv_dist[4] = {1.0, 1.0, 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0)};
    const double denom = sigma > 0.0 ? 2.0 * sigma * sigma : 0.0;

    double boundary = 0.0, max_wsum = 0.0;
    std::vector<double> wsum(static_cast<std::size_t>(N), 0.0);
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            const int p = r * C + c;
            for (int k = 0; k < n_off; ++k) {
                const int nr = r + dr[k], nc = c + dc[k];
                if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
                const int q = nr * C + nc;
                const double d = I[static_cast<std::size_t>(p)] - I[static_cast<std::size_t>(q)];
                const double w = denom > 0.0 ? lam * std::exp(-(d * d) / denom) * inv_dist[k]
                                             : (d == 0.0 ? lam * inv_dist[k] : 0.0);
                if (!(w > 1e-12)) continue;
                wsum[static_cast<std::size_t>(p)] += w;
                wsum[static_cast<std::size_t>(q)] += w;
                if ((mask.data[static_cast<std::size_t>(p)] >= 0.5f) !=
                    (mask.data[static_cast<std::size_t>(q)] >= 0.5f))
                    boundary += w;
            }
        }
    for (const double x : wsum) max_wsum = std::max(max_wsum, x);
    const double K = 1.0 + max_wsum;

    double data = 0.0;
    for (int i = 0; i < N; ++i) {
        const std::size_t u = static_cast<std::size_t>(i);
        const bool label_fg = mask.data[u] >= 0.5f;
        if (is_fg[u]) data += label_fg ? 0.0 : K;
        else if (is_bg[u]) data += label_fg ? K : 0.0;
        else {
            const double x = I[u];
            data += label_fg ? (x - mu_fg) * (x - mu_fg) : (x - mu_bg) * (x - mu_bg);
        }
    }
    return data + boundary;
}

// ---- Graph cut: golden segmentations ----

TEST(ImageGraphCut, StepImageCutsExactlyAtTheStep) {
    Image gray(8, 8, 1);
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c) gray.at(r, c, 0) = c < 4 ? 0.25f : 0.75f;
    const Image fg = gc_mask(8, 8, {{4, 6}});
    const Image bg = gc_mask(8, 8, {{4, 1}});

    const Image mask = graph_cut_segment(gray, fg, bg, 1.0, 0.1, 4);
    gc_expect_mask(mask, {"00001111", "00001111", "00001111", "00001111",
                          "00001111", "00001111", "00001111", "00001111"});

    // The 8 crossing edges each cost exp(-0.5^2 / (2*0.1^2)); every within-region
    // edge costs 1, and both regions' data terms vanish on their own side.
    EXPECT_NEAR(min_cut_value(gray, fg, bg, 1.0, 0.1, 4), 8.0 * std::exp(-12.5), 1e-12);
}

TEST(ImageGraphCut, LambdaZeroIsPerPixelThreshold) {
    // Dyadic ramp: every value is exact in float, so the tie at 0.5 is exact too.
    Image gray(3, 3, 1);
    const float ramp[9] = {0.f, 0.125f, 0.25f, 0.375f, 0.5f, 0.625f, 0.75f, 0.875f, 1.f};
    for (int i = 0; i < 9; ++i) gray.data[static_cast<std::size_t>(i)] = ramp[i];
    const Image fg = gc_mask(3, 3, {{2, 2}});
    const Image bg = gc_mask(3, 3, {{0, 0}});

    // mu_fg = 1, mu_bg = 0, threshold 0.5. The centre pixel sits exactly on the
    // threshold and goes to background by the tie rule.
    gc_expect_mask(graph_cut_segment(gray, fg, bg, 0.0, 0.1, 4), {"000", "001", "111"});

    // sum over the 7 unseeded pixels of min(D_fg, D_bg).
    EXPECT_DOUBLE_EQ(min_cut_value(gray, fg, bg, 0.0, 0.1, 4), 0.6875);
}

TEST(ImageGraphCut, UniformImageOneSeedEachSide) {
    const Image gray(4, 4, 1, 0.5f);
    const Image fg = gc_mask(4, 4, {{0, 0}});
    const Image bg = gc_mask(4, 4, {{3, 3}});

    // Every weight is 1 and every data term 0, so the cut is the cheapest edge
    // boundary separating the corners: isolating a degree-2 corner costs 2.
    // {(0,0)} and the complement of {(3,3)} both achieve it; the minimal source
    // side is their intersection.
    gc_expect_mask(graph_cut_segment(gray, fg, bg, 1.0, 1.0, 4),
                   {"1000", "0000", "0000", "0000"});
    EXPECT_DOUBLE_EQ(min_cut_value(gray, fg, bg, 1.0, 1.0, 4), 2.0);
}

TEST(ImageGraphCut, HandComputedThreeByThreeCutValue) {
    Image gray(3, 3, 1, 0.f);
    for (int r = 0; r < 3; ++r) gray.at(r, 2, 0) = 1.f;

    gc_expect_mask(graph_cut_segment(gray, 1.0, 1.0, 4), {"001", "001", "001"});

    // 2-means gives mu_bg = 0, mu_fg = 1; all data terms vanish on the chosen
    // labelling and the boundary is three edges of weight exp(-1/2). The runner
    // up (all background) costs 3.
    EXPECT_NEAR(min_cut_value(gray, 1.0, 1.0, 4), 3.0 * std::exp(-0.5), 1e-12);
}

TEST(ImageGraphCut, CenterSeedVersusCornerSeedFourConnected) {
    const Image gray(3, 3, 1, 0.5f);
    const Image fg = gc_mask(3, 3, {{1, 1}});
    const Image bg = gc_mask(3, 3, {{0, 0}});

    // Isolating the degree-2 corner beats isolating the degree-4 centre.
    gc_expect_mask(graph_cut_segment(gray, fg, bg, 1.0, 1.0, 4), {"011", "111", "111"});
    EXPECT_DOUBLE_EQ(min_cut_value(gray, fg, bg, 1.0, 1.0, 4), 2.0);
}

TEST(ImageGraphCut, EightConnectivityAddsDiagonalCost) {
    const Image gray(3, 3, 1, 0.5f);
    const Image fg = gc_mask(3, 3, {{1, 1}});
    const Image bg = gc_mask(3, 3, {{0, 0}});

    gc_expect_mask(graph_cut_segment(gray, fg, bg, 1.0, 1.0, 8), {"011", "111", "111"});
    // The corner gains a diagonal link to the centre, weighted lambda/sqrt(2):
    // this pins the 1/dist normalisation.
    EXPECT_NEAR(min_cut_value(gray, fg, bg, 1.0, 1.0, 8), 2.0 + 1.0 / std::sqrt(2.0), 1e-12);
}

TEST(ImageGraphCut, SymmetricCheckerboardLambdaZero) {
    Image gray(4, 4, 1);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) gray.at(r, c, 0) = ((r + c) % 2 == 0) ? 1.f : 0.f;

    // Each pixel sits exactly on one of the two means.
    gc_expect_mask(graph_cut_segment(gray, 0.0, 1.0, 4), {"1010", "0101", "1010", "0101"});
    EXPECT_DOUBLE_EQ(min_cut_value(gray, 0.0, 1.0, 4), 0.0);
}

TEST(ImageGraphCut, SymmetricCheckerboardStrongSmoothnessCollapsesToBackground) {
    Image gray(4, 4, 1);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) gray.at(r, c, 0) = ((r + c) % 2 == 0) ? 1.f : 0.f;

    // All-foreground and all-background both cost 8 while the checkerboard
    // labelling costs 24*10*exp(-0.5); the minimal-source-side rule picks the
    // empty one. Labelling by search-tree membership instead would flake here.
    gc_expect_mask(graph_cut_segment(gray, 10.0, 1.0, 4), {"0000", "0000", "0000", "0000"});
    EXPECT_DOUBLE_EQ(min_cut_value(gray, 10.0, 1.0, 4), 8.0);
}

TEST(ImageGraphCut, AllForegroundSeedsAndAllBackgroundSeeds) {
    const Image gray(3, 3, 1, 0.3f);
    const Image all_on(3, 3, 1, 1.f);

    gc_expect_mask(graph_cut_segment(gray, all_on, Image{}, 1.0, 0.1, 4), {"111", "111", "111"});
    EXPECT_DOUBLE_EQ(min_cut_value(gray, all_on, Image{}, 1.0, 0.1, 4), 0.0);

    gc_expect_mask(graph_cut_segment(gray, Image{}, all_on, 1.0, 0.1, 4), {"000", "000", "000"});
    EXPECT_DOUBLE_EQ(min_cut_value(gray, Image{}, all_on, 1.0, 0.1, 4), 0.0);
}

TEST(ImageGraphCut, SingleRowSingleColumnAndOneByOne) {
    const float vals[6] = {0.f, 0.f, 0.f, 0.f, 1.f, 1.f};

    Image row(1, 6, 1);
    for (int i = 0; i < 6; ++i) row.data[static_cast<std::size_t>(i)] = vals[i];
    gc_expect_mask(graph_cut_segment(row, 1.0, 0.25, 4), {"000011"});
    EXPECT_NEAR(min_cut_value(row, 1.0, 0.25, 4), std::exp(-8.0), 1e-15);

    // No diagonal fits in a single row, so connectivity 8 is byte-identical.
    EXPECT_EQ(graph_cut_segment(row, 1.0, 0.25, 8).data, graph_cut_segment(row, 1.0, 0.25, 4).data);
    EXPECT_DOUBLE_EQ(min_cut_value(row, 1.0, 0.25, 8), min_cut_value(row, 1.0, 0.25, 4));

    Image col(6, 1, 1);
    for (int i = 0; i < 6; ++i) col.data[static_cast<std::size_t>(i)] = vals[i];
    gc_expect_mask(graph_cut_segment(col, 1.0, 0.25, 4), {"0", "0", "0", "0", "1", "1"});
    EXPECT_NEAR(min_cut_value(col, 1.0, 0.25, 4), std::exp(-8.0), 1e-15);

    const Image one(1, 1, 1, 0.7f);
    const Image on(1, 1, 1, 1.f);
    gc_expect_mask(graph_cut_segment(one, on, Image{}, 1.0, 0.1, 4), {"1"});
    gc_expect_mask(graph_cut_segment(one, Image{}, on, 1.0, 0.1, 4), {"0"});
    gc_expect_mask(graph_cut_segment(one, 1.0, 0.1, 4), {"0"});
    EXPECT_DOUBLE_EQ(min_cut_value(one, on, Image{}, 1.0, 0.1, 4), 0.0);
    EXPECT_DOUBLE_EQ(min_cut_value(one, Image{}, on, 1.0, 0.1, 4), 0.0);
    EXPECT_DOUBLE_EQ(min_cut_value(one, 1.0, 0.1, 4), 0.0);
}

TEST(ImageGraphCut, SigmaZeroIsHardPotts) {
    Image gray(3, 4, 1);
    for (int r = 0; r < 3; ++r) {
        gray.at(r, 0, 0) = 0.f;
        gray.at(r, 1, 0) = 0.f;
        gray.at(r, 2, 0) = 0.5f;
        gray.at(r, 3, 0) = 1.f;
    }

    // sigma = 0 links only exactly-equal neighbours, so the graph splits into
    // three components (cols 0-1, col 2, col 3) with no edges between them and
    // each takes its cheaper label independently. 2-means gives mu_bg = 1/6,
    // mu_fg = 1, so the value is 6*(1/6)^2 + 3*(1/3)^2 = 1/6 + 1/3.
    gc_expect_mask(graph_cut_segment(gray, 1.0, 0.0, 4), {"0001", "0001", "0001"});
    EXPECT_NEAR(min_cut_value(gray, 1.0, 0.0, 4), 0.5, 1e-12);
}

TEST(ImageGraphCut, SeedsOverrideIntensityOrder) {
    Image gray(4, 4, 1);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) gray.at(r, c, 0) = c < 2 ? 0.f : 1.f;
    // The foreground seed sits on the *dark* half.
    const Image fg = gc_mask(4, 4, {{0, 0}});
    const Image bg = gc_mask(4, 4, {{0, 3}});

    gc_expect_mask(graph_cut_segment(gray, fg, bg, 1.0, 0.2, 4),
                   {"1100", "1100", "1100", "1100"});
    EXPECT_NEAR(min_cut_value(gray, fg, bg, 1.0, 0.2, 4), 4.0 * std::exp(-12.5), 1e-12);
}

TEST(ImageGraphCut, SeedOverlapPrefersForeground) {
    const Image gray(3, 3, 1, 0.5f);
    const Image fg = gc_mask(3, 3, {{1, 1}});
    const Image bg = gc_mask(3, 3, {{1, 1}, {0, 0}});  // (1,1) is in both masks

    // Identical to the single-seed case: foreground wins the overlap.
    gc_expect_mask(graph_cut_segment(gray, fg, bg, 1.0, 1.0, 4), {"011", "111", "111"});
    EXPECT_DOUBLE_EQ(min_cut_value(gray, fg, bg, 1.0, 1.0, 4), 2.0);
}

TEST(ImageGraphCut, EightConnectedStepAddsDiagonalBoundary) {
    Image gray(6, 6, 1);
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c) gray.at(r, c, 0) = c < 3 ? 0.25f : 0.75f;
    const Image fg = gc_mask(6, 6, {{3, 4}});
    const Image bg = gc_mask(6, 6, {{3, 1}});

    gc_expect_mask(graph_cut_segment(gray, fg, bg, 1.0, 0.1, 4),
                   {"000111", "000111", "000111", "000111", "000111", "000111"});
    EXPECT_NEAR(min_cut_value(gray, fg, bg, 1.0, 0.1, 4), 6.0 * std::exp(-12.5), 1e-12);

    // Same boundary, but 10 diagonal crossings at 1/sqrt(2) join the 6 axial ones.
    gc_expect_mask(graph_cut_segment(gray, fg, bg, 1.0, 0.1, 8),
                   {"000111", "000111", "000111", "000111", "000111", "000111"});
    EXPECT_NEAR(min_cut_value(gray, fg, bg, 1.0, 0.1, 8),
                std::exp(-12.5) * (6.0 + 10.0 / std::sqrt(2.0)), 1e-12);
}

TEST(ImageGraphCut, BlobGolden) {
    Image gray(16, 16, 1, 0.2f);
    for (int r = 4; r < 12; ++r)
        for (int c = 4; c < 12; ++c) gray.at(r, c, 0) = 0.8f;
    const Image fg = gc_mask(16, 16, {{8, 8}});
    const Image bg = gc_mask(16, 16, {{0, 0}});

    const Image mask = graph_cut_segment(gray, fg, bg, 1.0, 0.1, 4);
    ASSERT_EQ(mask.rows, 16);
    ASSERT_EQ(mask.cols, 16);
    ASSERT_EQ(mask.channels, 1);
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c) {
            const bool want = r >= 4 && r < 12 && c >= 4 && c < 12;
            EXPECT_FLOAT_EQ(mask.at(r, c, 0), want ? 1.f : 0.f) << "at (" << r << "," << c << ")";
        }
    EXPECT_EQ(gc_count_fg(mask), 64);

    // The boundary is 32 axial edges. The tolerance absorbs 0.2f/0.8f not being
    // exactly representable, so the measured contrast is 0.60000002, not 0.6.
    EXPECT_NEAR(min_cut_value(gray, fg, bg, 1.0, 0.1, 4), 32.0 * std::exp(-18.0), 1e-11);
}

TEST(ImageGraphCut, RgbInputMatchesRgb2Gray) {
    Image rgb(3, 3, 3, 0.f);
    for (int r = 0; r < 3; ++r)
        for (int ch = 0; ch < 3; ++ch) rgb.at(r, 2, ch) = 1.f;
    const Image gray = rgb2gray(rgb);

    const Image from_rgb = graph_cut_segment(rgb, 1.0, 1.0, 4);
    const Image from_gray = graph_cut_segment(gray, 1.0, 1.0, 4);
    gc_expect_mask(from_rgb, {"001", "001", "001"});
    EXPECT_EQ(from_rgb.data, from_gray.data);
    EXPECT_DOUBLE_EQ(min_cut_value(rgb, 1.0, 1.0, 4), min_cut_value(gray, 1.0, 1.0, 4));

    // A 2-channel image uses channel 0 -- rgb2gray() would read past its end.
    Image two(3, 3, 2, 0.f);
    for (int r = 0; r < 3; ++r) two.at(r, 2, 0) = 1.f;
    gc_expect_mask(graph_cut_segment(two, 1.0, 1.0, 4), {"001", "001", "001"});
    EXPECT_NEAR(min_cut_value(two, 1.0, 1.0, 4), 3.0 * std::exp(-0.5), 1e-12);
}

TEST(ImageGraphCut, MultiChannelSeedMasksMarkPixels) {
    const Image gray(3, 3, 1, 0.5f);
    Image fg(3, 3, 3, 0.f), bg(3, 3, 3, 0.f);
    for (int ch = 0; ch < 3; ++ch) {
        fg.at(1, 1, ch) = 1.f;
        bg.at(0, 0, ch) = 1.f;
    }
    // An all-white (1,1,1) mask pixel reads as 1.0 and marks the pixel, so this
    // matches the single-channel seeding of the same two pixels.
    gc_expect_mask(graph_cut_segment(gray, fg, bg, 1.0, 1.0, 4), {"011", "111", "111"});
    EXPECT_DOUBLE_EQ(min_cut_value(gray, fg, bg, 1.0, 1.0, 4), 2.0);
}

TEST(ImageGraphCut, UnseededOverloadMatchesEmptySeedMasks) {
    Image gray(4, 4, 1);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            gray.at(r, c, 0) = static_cast<float>(((r * 4 + c) % 7) / 8.0);

    EXPECT_EQ(graph_cut_segment(gray).data, graph_cut_segment(gray, Image{}, Image{}).data);
    EXPECT_DOUBLE_EQ(min_cut_value(gray), min_cut_value(gray, Image{}, Image{}));
    EXPECT_EQ(graph_cut_segment(gray, 1.0, 0.3, 8).data,
              graph_cut_segment(gray, Image{}, Image{}, 1.0, 0.3, 8).data);

    // A seed mask present but marking nothing behaves exactly like an empty one.
    const Image blank(4, 4, 1, 0.f);
    EXPECT_EQ(graph_cut_segment(gray, blank, blank, 1.0, 0.3, 4).data,
              graph_cut_segment(gray, 1.0, 0.3, 4).data);
    EXPECT_DOUBLE_EQ(min_cut_value(gray, blank, blank, 1.0, 0.3, 4),
                     min_cut_value(gray, 1.0, 0.3, 4));
}

TEST(ImageGraphCut, MinCutValueEqualsEnergyOfReturnedMask) {
    Image gray(4, 4, 1);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            gray.at(r, c, 0) = static_cast<float>(((r * 4 + c) % 7) / 8.0);
    const Image fg = gc_mask(4, 4, {{0, 0}});
    const Image bg = gc_mask(4, 4, {{3, 3}});
    const Image all_fg(4, 4, 1, 1.f), all_bg(4, 4, 1, 0.f);

    const double lambdas[4] = {0.0, 0.25, 1.0, 4.0};
    const double sigmas[4] = {0.0, 0.05, 0.3, 1.5};
    const int conns[2] = {4, 8};

    for (const double lambda : lambdas)
        for (const double sigma : sigmas)
            for (const int conn : conns)
                for (int seeded = 0; seeded < 2; ++seeded) {
                    const Image f = seeded ? fg : Image{};
                    const Image b = seeded ? bg : Image{};
                    const Image mask = graph_cut_segment(gray, f, b, lambda, sigma, conn);
                    const double value = min_cut_value(gray, f, b, lambda, sigma, conn);
                    SCOPED_TRACE(testing::Message() << "lambda=" << lambda << " sigma=" << sigma
                                                    << " conn=" << conn << " seeded=" << seeded);
                    EXPECT_NEAR(value, gc_energy_of(gray, f, b, lambda, sigma, conn, mask), 1e-9);
                    // The cut is a global optimum, so no fixed labelling beats it.
                    EXPECT_LE(value, gc_energy_of(gray, f, b, lambda, sigma, conn, all_fg) + 1e-9);
                    EXPECT_LE(value, gc_energy_of(gray, f, b, lambda, sigma, conn, all_bg) + 1e-9);
                }
}

TEST(ImageGraphCut, DegenerateInputsReturnEmptyOrZero) {
    EXPECT_TRUE(graph_cut_segment(Image{}, Image{}, Image{}).empty());
    EXPECT_TRUE(graph_cut_segment(Image{}).empty());
    EXPECT_DOUBLE_EQ(min_cut_value(Image{}), 0.0);
    EXPECT_DOUBLE_EQ(min_cut_value(Image{}, Image{}, Image{}), 0.0);

    const Image gray(3, 3, 1, 0.5f);
    EXPECT_TRUE(graph_cut_segment(gray, Image(2, 2, 1, 1.f), Image{}).empty());
    EXPECT_DOUBLE_EQ(min_cut_value(gray, Image(2, 2, 1, 1.f), Image{}), 0.0);
    EXPECT_TRUE(graph_cut_segment(gray, Image{}, Image(4, 4, 1, 1.f)).empty());
    EXPECT_DOUBLE_EQ(min_cut_value(gray, Image{}, Image(4, 4, 1, 1.f)), 0.0);

    // Uniform and unseeded: both means coincide, every tie goes to background.
    const Image uniform(4, 4, 1, 0.42f);
    gc_expect_mask(graph_cut_segment(uniform, 1.0, 0.1, 4), {"0000", "0000", "0000", "0000"});
    EXPECT_DOUBLE_EQ(min_cut_value(uniform, 1.0, 0.1, 4), 0.0);

    const Image fg = gc_mask(3, 3, {{2, 2}});
    const Image bg = gc_mask(3, 3, {{0, 0}});
    // A negative lambda is the same as dropping the pairwise term.
    EXPECT_EQ(graph_cut_segment(gray, fg, bg, -5.0, 0.1, 4).data,
              graph_cut_segment(gray, fg, bg, 0.0, 0.1, 4).data);
    EXPECT_DOUBLE_EQ(min_cut_value(gray, fg, bg, -5.0, 0.1, 4),
                     min_cut_value(gray, fg, bg, 0.0, 0.1, 4));
    // Any connectivity other than 8 is treated as 4.
    EXPECT_EQ(graph_cut_segment(gray, fg, bg, 1.0, 0.1, 7).data,
              graph_cut_segment(gray, fg, bg, 1.0, 0.1, 4).data);
    EXPECT_DOUBLE_EQ(min_cut_value(gray, fg, bg, 1.0, 0.1, 7),
                     min_cut_value(gray, fg, bg, 1.0, 0.1, 4));

    // The output is always rows x cols x 1 and strictly binary.
    const Image mask = graph_cut_segment(gray, fg, bg, 1.0, 0.1, 8);
    EXPECT_EQ(mask.rows, 3);
    EXPECT_EQ(mask.cols, 3);
    EXPECT_EQ(mask.channels, 1);
    for (const float v : mask.data) EXPECT_TRUE(v == 0.f || v == 1.f);
}

TEST(ImageGraphCut, DeterministicRepeatedCalls) {
    Image gray(9, 9, 1);
    for (int r = 0; r < 9; ++r)
        for (int c = 0; c < 9; ++c)
            gray.at(r, c, 0) = 0.05f * static_cast<float>((7 * r + 3 * c) % 11);

    const Image a = graph_cut_segment(gray, 1.0, 0.25, 8);
    const Image b = graph_cut_segment(gray, 1.0, 0.25, 8);
    EXPECT_EQ(a.data, b.data);
    EXPECT_DOUBLE_EQ(min_cut_value(gray, 1.0, 0.25, 8), min_cut_value(gray, 1.0, 0.25, 8));
    EXPECT_GT(min_cut_value(gray, 1.0, 0.25, 8), 0.0);
}

// ---- GrabCut ----

static Image gc_blob8() {
    Image gray(8, 8, 1, 0.2f);
    for (int r = 2; r < 6; ++r)
        for (int c = 2; c < 6; ++c) gray.at(r, c, 0) = 0.85f;
    return gray;
}

TEST(ImageGrabCut, RectangleAroundABlobRecoversTheBlob) {
    const Image gray = gc_blob8();
    const std::initializer_list<const char*> want = {"00000000", "00000000", "00111100",
                                                     "00111100", "00111100", "00111100",
                                                     "00000000", "00000000"};

    gc_expect_mask(grabcut_segment(gray, 1, 1, 7, 7, 5, 3, 1.0, 0.1, 4), want);
    gc_expect_mask(grabcut_segment(gray, 1, 1, 7, 7, 5, 3, 1.0, 0.1, 8), want);
    gc_expect_mask(grabcut_segment(gray, 1, 1, 7, 7, 5, 3, 0.0, 0.1, 4), want);
    gc_expect_mask(grabcut_segment(gray, 1, 1, 7, 7, 5, 3, 1.0, 0.0, 4), want);
    gc_expect_mask(grabcut_segment(gray, 1, 1, 7, 7, 5, 3, 1.0, 0.1, 13), want);  // conn -> 4
    gc_expect_mask(grabcut_segment(gray, 1, 1, 7, 7, 5, 1, 1.0, 0.1, 4), want);
    gc_expect_mask(grabcut_segment(gray, 1, 1, 7, 7, 5, 8, 1.0, 0.1, 4), want);
    gc_expect_mask(grabcut_segment(gray, 1, 1, 7, 7, 5, 0, 1.0, 0.1, 4), want);   // K -> 1
    gc_expect_mask(grabcut_segment(gray, 1, 1, 7, 7, 5, 99, 1.0, 0.1, 4), want);  // K -> 8
    gc_expect_mask(grabcut_segment(gray, 1, 1, 7, 7, 1, 3, 1.0, 0.1, 4), want);
    gc_expect_mask(grabcut_segment(gray, 1, 1, 7, 7, 0, 3, 1.0, 0.1, 4), want);   // iters -> 1

    Image rgb(8, 8, 3, 0.f);
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            for (int ch = 0; ch < 3; ++ch) rgb.at(r, c, ch) = gray.at(r, c, 0);
    gc_expect_mask(grabcut_segment(rgb, 1, 1, 7, 7, 5, 3, 1.0, 0.1, 4), want);
}

TEST(ImageGrabCut, DegenerateRectanglesAndImages) {
    EXPECT_TRUE(grabcut_segment(Image{}, 0, 0, 2, 2).empty());

    // A rectangle that clamps to zero area gives an all-background mask.
    const Image uniform6(6, 6, 1, 0.5f);
    const Image empty_rect = grabcut_segment(uniform6, 3, 3, 1, 1);
    EXPECT_EQ(empty_rect.rows, 6);
    EXPECT_EQ(empty_rect.cols, 6);
    EXPECT_EQ(empty_rect.channels, 1);
    EXPECT_EQ(gc_count_fg(empty_rect), 0);

    // ... and so does one that is empty in only ONE dimension, which is the case the
    // line above cannot see. `r1 <= r0 || c1 <= c0` and `r1 <= r0 && c1 <= c0` agree
    // on a rectangle degenerate in both, and differ on every rectangle with zero
    // height and positive width, or the other way about. A zero-area rectangle has no
    // pixels in it whichever side is flat, and the blob image is used rather than a
    // uniform one so the two readings cannot agree by the data being featureless.
    const Image blob = gc_blob8();
    const Image flat_rows = grabcut_segment(blob, 4, 1, 4, 7);
    EXPECT_EQ(flat_rows.rows, 8);
    EXPECT_EQ(flat_rows.cols, 8);
    EXPECT_EQ(gc_count_fg(flat_rows), 0) << "a rectangle of zero height selected pixels";
    const Image flat_cols = grabcut_segment(blob, 1, 4, 7, 4);
    EXPECT_EQ(gc_count_fg(flat_cols), 0) << "a rectangle of zero width selected pixels";
    // And the same through the clamp rather than as written: c1 clamps to 0.
    EXPECT_EQ(gc_count_fg(grabcut_segment(blob, 1, -3, 7, -1)), 0);

    // A rectangle covering the whole image leaves no background samples for the
    // first fit, so the initial labelling stands.
    const Image gray = gc_blob8();
    EXPECT_EQ(gc_count_fg(grabcut_segment(gray, 0, 0, 8, 8)), 64);
    EXPECT_EQ(gc_count_fg(grabcut_segment(gray, -5, -5, 100, 100)), 64);

    // Uniform image: two identical mixtures, an uninformative data term.
    const Image uniform(6, 6, 1, 0.3f);
    EXPECT_EQ(gc_count_fg(grabcut_segment(uniform, 1, 1, 5, 5)), 0);

    const Image one(1, 1, 1, 0.5f);
    gc_expect_mask(grabcut_segment(one, 0, 0, 1, 1), {"1"});
}

TEST(ImageGrabCut, OneDimensionalAndLargerBlobs) {
    Image row(1, 8, 1, 0.1f);
    for (int c = 3; c < 6; ++c) row.at(0, c, 0) = 0.9f;
    gc_expect_mask(grabcut_segment(row, 0, 1, 1, 7), {"00011100"});

    Image gray(10, 10, 1, 0.15f);
    for (int r = 3; r < 8; ++r)
        for (int c = 3; c < 8; ++c) gray.at(r, c, 0) = 0.8f;
    const Image a = grabcut_segment(gray, 1, 1, 9, 9);
    for (int r = 0; r < 10; ++r)
        for (int c = 0; c < 10; ++c) {
            const bool want = r >= 3 && r < 8 && c >= 3 && c < 8;
            EXPECT_FLOAT_EQ(a.at(r, c, 0), want ? 1.f : 0.f) << "at (" << r << "," << c << ")";
        }
    EXPECT_EQ(gc_count_fg(a), 25);
    EXPECT_EQ(a.data, grabcut_segment(gray, 1, 1, 9, 9).data);  // deterministic

    // Three intensity levels: the mixture keeps both bright levels together
    // where a single foreground mean would split them.
    Image tri(12, 12, 1, 0.1f);
    for (int r = 2; r < 10; ++r)
        for (int c = 2; c < 10; ++c) tri.at(r, c, 0) = 0.6f;
    for (int r = 4; r < 8; ++r)
        for (int c = 4; c < 8; ++c) tri.at(r, c, 0) = 0.95f;
    const Image t = grabcut_segment(tri, 1, 1, 11, 11, 5, 3, 1.0, 0.15, 8);
    for (int r = 0; r < 12; ++r)
        for (int c = 0; c < 12; ++c) {
            const bool want = r >= 2 && r < 10 && c >= 2 && c < 10;
            EXPECT_FLOAT_EQ(t.at(r, c, 0), want ? 1.f : 0.f) << "at (" << r << "," << c << ")";
        }
    EXPECT_EQ(gc_count_fg(t), 64);
}
