// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#define _USE_MATH_DEFINES
#include "ms/image/image.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms::image;

// ---- fixtures ----

// Bright 20x20 square at rows/cols [10, 30) on a 40x40 dark canvas.
static Image make_square40() {
    Image img(40, 40, 1, 0.f);
    for (int r = 10; r < 30; ++r)
        for (int c = 10; c < 30; ++c) img.at(r, c, 0) = 1.f;
    return img;
}

// 8x8-block checkerboard on 64x64.
static Image make_checkerboard64() {
    Image img(64, 64, 1, 0.f);
    for (int r = 0; r < 64; ++r)
        for (int c = 0; c < 64; ++c)
            img.at(r, c, 0) = (((r / 8) + (c / 8)) % 2 == 0) ? 1.f : 0.f;
    return img;
}

// Deterministic texture: SplitMix64 white noise, blurred, min-max stretched.
// Uses no std::mt19937, whose distribution objects are not portable.
static Image make_texture(int side) {
    std::uint64_t s = 12345ull;
    auto nxt = [&s]() {
        s += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    };
    Image g(side, side, 1, 0.f);
    for (int r = 0; r < side; ++r)
        for (int c = 0; c < side; ++c)
            g.at(r, c, 0) = static_cast<float>(
                (static_cast<double>(nxt() >> 11) + 0.5) * (1.0 / 9007199254740992.0));
    g = imgaussfilt(g, 1.2f);
    float lo = 1e9f, hi = -1e9f;
    for (float v : g.data) { lo = std::min(lo, v); hi = std::max(hi, v); }
    for (float& v : g.data) v = (v - lo) / (hi - lo);
    return g;
}

// Isotropic Gaussian blob, amplitude 1, on a black canvas.
static Image make_blob(int side, double cx, double cy, double s) {
    Image g(side, side, 1, 0.f);
    for (int r = 0; r < side; ++r)
        for (int c = 0; c < side; ++c) {
            const double dx = c - cx, dy = r - cy;
            g.at(r, c, 0) = static_cast<float>(std::exp(-(dx * dx + dy * dy) / (2.0 * s * s)));
        }
    return g;
}

// A 4x4 grid of Gaussian blobs of four different widths and alternating sign
// on a mid-grey canvas -- the stimulus a DoG detector is built for, and far
// richer than make_texture() (whose energy sits below SIFT's 1.6 base blur).
static Image make_blob_field(int side) {
    Image g(side, side, 1, 0.f);
    const double sig[4] = {2.5, 3.0, 4.0, 5.0};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            const double cx = 16.0 + 32.0 * j, cy = 16.0 + 32.0 * i;
            const double s = sig[static_cast<std::size_t>((i + j) % 4)];
            const double amp = ((i + j) % 2 == 0) ? 1.0 : -1.0;
            for (int r = 0; r < side; ++r)
                for (int c = 0; c < side; ++c) {
                    const double dx = c - cx, dy = r - cy;
                    const double e = (dx * dx + dy * dy) / (2.0 * s * s);
                    if (e < 25.0) g.at(r, c, 0) += static_cast<float>(amp * 0.5 * std::exp(-e));
                }
        }
    for (float& v : g.data) v = std::clamp(v + 0.5f, 0.f, 1.f);
    return g;
}

static bool has_corner(const std::vector<KeyPoint>& kps, float y, float x) {
    for (const auto& k : kps)
        if (k.y == y && k.x == x) return true;
    return false;
}

// ---- FAST ----

TEST(ImageFast, BrightSquareYieldsExactlyFourCorners) {
    const auto kps = fast_corners(make_square40(), 0.05f, true);
    ASSERT_EQ(kps.size(), 4u);
    const float ey[4] = {10.f, 10.f, 29.f, 29.f};
    const float ex[4] = {10.f, 29.f, 10.f, 29.f};
    for (int i = 0; i < 4; ++i) {
        const auto& k = kps[static_cast<std::size_t>(i)];
        EXPECT_FLOAT_EQ(k.y, ey[i]) << "corner " << i;   // raster order
        EXPECT_FLOAT_EQ(k.x, ex[i]) << "corner " << i;
        // 11 arc pixels, each contributing (1.0 - 0.05).
        EXPECT_NEAR(10.45f, k.response, 1e-3f);
    }
}

TEST(ImageFast, CheckerboardYieldsNoCorners) {
    // FAST-9 cannot fire on an X-junction: the four ~4-pixel arcs a
    // checkerboard corner produces never reach the required 9.
    EXPECT_EQ(fast_corners(make_checkerboard64(), 0.05f, true).size(), 0u);
    EXPECT_EQ(fast_corners(make_checkerboard64(), 0.05f, false).size(), 0u);
}

TEST(ImageFast, ConstantImageYieldsNoCorners) {
    EXPECT_EQ(fast_corners(Image(20, 20, 1, 0.5f)).size(), 0u);
    EXPECT_EQ(fast_corners(Image(20, 20, 1, 0.f)).size(), 0u);
    EXPECT_EQ(fast_corners(Image(20, 20, 1, 1.f)).size(), 0u);
    // A negative threshold clamps to 0, and the arc tests are strict, so a
    // constant image still yields nothing.
    EXPECT_EQ(fast_corners(Image(20, 20, 1, 0.5f), -1.f, true).size(), 0u);
}

TEST(ImageFast, DegenerateSizes) {
    EXPECT_EQ(fast_corners(Image{}).size(), 0u);
    EXPECT_EQ(fast_corners(Image(1, 1, 1, 1.f)).size(), 0u);
    EXPECT_EQ(fast_corners(Image(6, 6, 1, 1.f)).size(), 0u);
    EXPECT_EQ(fast_corners(Image(7, 6, 1, 1.f)).size(), 0u);
    EXPECT_EQ(fast_corners(Image(6, 7, 1, 1.f)).size(), 0u);
}

TEST(ImageFast, DiamondTipsAreDetected) {
    Image d(41, 41, 1, 0.f);
    for (int r = 0; r < 41; ++r)
        for (int c = 0; c < 41; ++c)
            if (std::abs(r - 20) + std::abs(c - 20) <= 12) d.at(r, c, 0) = 1.f;
    const auto kps = fast_corners(d, 0.05f, true);
    ASSERT_EQ(kps.size(), 4u);
    EXPECT_TRUE(has_corner(kps, 8.f, 20.f));
    EXPECT_TRUE(has_corner(kps, 20.f, 8.f));
    EXPECT_TRUE(has_corner(kps, 20.f, 31.f));
    EXPECT_TRUE(has_corner(kps, 31.f, 20.f));
    for (const auto& k : kps) EXPECT_NEAR(10.45f, k.response, 1e-3f);
}

TEST(ImageFast, LShapeYieldsConvexAndConcaveCorners) {
    Image l = make_square40();
    for (int r = 10; r < 20; ++r)
        for (int c = 20; c < 30; ++c) l.at(r, c, 0) = 0.f;
    const auto kps = fast_corners(l, 0.05f, true);
    // Five convex corners plus the one concave (reflex) corner of the notch.
    ASSERT_EQ(kps.size(), 6u);
    EXPECT_TRUE(has_corner(kps, 10.f, 10.f));
    EXPECT_TRUE(has_corner(kps, 10.f, 19.f));
    EXPECT_TRUE(has_corner(kps, 19.f, 20.f));
    EXPECT_TRUE(has_corner(kps, 20.f, 29.f));
    EXPECT_TRUE(has_corner(kps, 29.f, 10.f));
    EXPECT_TRUE(has_corner(kps, 29.f, 29.f));
    for (const auto& k : kps) EXPECT_NEAR(10.45f, k.response, 1e-3f);
}

TEST(ImageFast, ThresholdReducesCandidates) {
    const Image img = make_square40();
    const auto lo = fast_corners(img, 0.05f, false);
    const auto mid = fast_corners(img, 0.5f, false);
    const auto hi = fast_corners(img, 1.0f, false);
    EXPECT_GE(lo.size(), mid.size());
    EXPECT_GE(mid.size(), hi.size());
    // Pixels are exactly 0 or 1, so every threshold below 1 selects the same
    // 24 candidates and only the score changes; at 1.0 the strict tests fail.
    EXPECT_EQ(lo.size(), 24u);
    EXPECT_EQ(mid.size(), 24u);
    EXPECT_EQ(hi.size(), 0u);
    const auto strong = fast_corners(img, 0.5f, true);
    ASSERT_EQ(strong.size(), 4u);
    for (const auto& k : strong) EXPECT_NEAR(5.5f, k.response, 1e-3f);  // 11 * 0.5
}

TEST(ImageFast, NonMaxSuppressionThinsClusters) {
    const Image img = make_square40();
    const auto kept = fast_corners(img, 0.05f, true);
    const auto all = fast_corners(img, 0.05f, false);
    EXPECT_EQ(kept.size(), 4u);
    EXPECT_EQ(all.size(), 24u);
    for (const auto& k : kept) EXPECT_TRUE(has_corner(all, k.y, k.x));
    // Raster order is preserved by both passes.
    for (std::size_t i = 1; i < all.size(); ++i)
        EXPECT_TRUE(all[i - 1].y < all[i].y || (all[i - 1].y == all[i].y && all[i - 1].x < all[i].x));
}

TEST(ImageFast, MultiChannelInputMatchesGrayscale) {
    const Image gray = make_square40();
    const auto base = fast_corners(gray, 0.05f, true);
    const auto rgb = fast_corners(gray2rgb(gray), 0.05f, true);
    ASSERT_EQ(rgb.size(), base.size());
    for (std::size_t i = 0; i < base.size(); ++i) {
        EXPECT_FLOAT_EQ(rgb[i].x, base[i].x);
        EXPECT_FLOAT_EQ(rgb[i].y, base[i].y);
        EXPECT_NEAR(rgb[i].response, base[i].response, 1e-4f);
    }
    // A 2-channel image uses channel 0 (rgb2gray would read out of bounds).
    Image two(40, 40, 2, 0.f);
    for (int r = 0; r < 40; ++r)
        for (int c = 0; c < 40; ++c) {
            two.at(r, c, 0) = gray.at(r, c, 0);
            two.at(r, c, 1) = 0.5f;
        }
    const auto twoch = fast_corners(two, 0.05f, true);
    ASSERT_EQ(twoch.size(), base.size());
    for (std::size_t i = 0; i < base.size(); ++i) {
        EXPECT_FLOAT_EQ(twoch[i].x, base[i].x);
        EXPECT_FLOAT_EQ(twoch[i].y, base[i].y);
    }
}

// ---- ORB ----

TEST(ImageOrb, PatternIsDeterministicAndValid) {
    const auto& p = orb_sampling_pattern();
    EXPECT_EQ(&p, &orb_sampling_pattern());  // one shared table

    const std::array<int, 4> p0 = {-2, 11, -3, -9};
    const std::array<int, 4> p1 = {7, -2, 4, -7};
    const std::array<int, 4> p2 = {1, 3, 3, 2};
    const std::array<int, 4> p255 = {6, 8, -7, 6};
    EXPECT_EQ(p[0], p0);
    EXPECT_EQ(p[1], p1);
    EXPECT_EQ(p[2], p2);
    EXPECT_EQ(p[255], p255);

    long sum_abs = 0, sum_sq = 0;
    int degenerate = 0;
    for (const auto& q : p) {
        for (int t = 0; t < 4; ++t) {
            const int v = q[static_cast<std::size_t>(t)];
            EXPECT_GE(v, -15);
            EXPECT_LE(v, 15);
            sum_abs += std::abs(v);
            sum_sq += static_cast<long>(v) * v;
        }
        if (q[0] == q[2] && q[1] == q[3]) ++degenerate;
    }
    EXPECT_EQ(sum_abs, 5099);
    EXPECT_EQ(sum_sq, 38705);
    EXPECT_EQ(degenerate, 0);
}

TEST(ImageOrb, OrientationPatchCoversSevenHundredNinePixels) {
    // Row extents of the radius-15 disc used by the intensity centroid.
    const int umax[16] = {15, 14, 14, 14, 14, 14, 13, 13, 12, 12, 11, 10, 9, 7, 5, 0};
    int area = 31;
    for (int k = 1; k < 16; ++k) {
        EXPECT_EQ(umax[k], static_cast<int>(std::sqrt(225.0 - static_cast<double>(k * k))));
        area += 2 * (2 * umax[k] + 1);
    }
    EXPECT_EQ(area, 709);
}

TEST(ImageOrb, DegenerateInputs) {
    EXPECT_TRUE(orb_detect_and_compute(Image{}).keypoints.empty());
    EXPECT_TRUE(orb_detect_and_compute(Image(20, 20, 1, 0.5f)).keypoints.empty());
    EXPECT_TRUE(orb_detect_and_compute(Image(40, 40, 1, 0.5f)).keypoints.empty());
    const auto none = orb_detect_and_compute(make_texture(96), 0);
    EXPECT_TRUE(none.keypoints.empty());
    EXPECT_TRUE(none.descriptors.empty());
    // A degenerate scale factor is clamped rather than rejected.
    EXPECT_FALSE(orb_detect_and_compute(make_texture(96), 50, 0.05f, 0, 0.5f).keypoints.empty());
}

TEST(ImageOrb, KeypointsAndDescriptorsAreParallelAndSorted) {
    const auto f = orb_detect_and_compute(make_texture(96), 200);
    ASSERT_EQ(f.keypoints.size(), f.descriptors.size());
    EXPECT_GE(f.keypoints.size(), 100u);
    EXPECT_LE(f.keypoints.size(), 200u);
    for (std::size_t i = 0; i < f.keypoints.size(); ++i) {
        const auto& k = f.keypoints[i];
        if (i > 0) {
            EXPECT_LE(k.response, f.keypoints[i - 1].response);
        }
        EXPECT_GE(k.octave, 0);
        EXPECT_GE(k.x, 0.f);
        EXPECT_LE(k.x, 95.f);
        EXPECT_GE(k.y, 0.f);
        EXPECT_LE(k.y, 95.f);
        EXPECT_GE(k.scale, 1.f);
        EXPECT_GE(k.orientation, -static_cast<float>(M_PI) - 1e-5f);
        EXPECT_LE(k.orientation, static_cast<float>(M_PI) + 1e-5f);
        EXPECT_GT(k.response, 0.f);
    }
}

TEST(ImageOrb, IsDeterministic) {
    const auto a = orb_detect_and_compute(make_texture(96), 200);
    const auto b = orb_detect_and_compute(make_texture(96), 200);
    ASSERT_EQ(a.keypoints.size(), b.keypoints.size());
    ASSERT_EQ(a.descriptors.size(), b.descriptors.size());
    for (std::size_t i = 0; i < a.keypoints.size(); ++i) {
        EXPECT_FLOAT_EQ(a.keypoints[i].x, b.keypoints[i].x);
        EXPECT_FLOAT_EQ(a.keypoints[i].y, b.keypoints[i].y);
        EXPECT_FLOAT_EQ(a.keypoints[i].scale, b.keypoints[i].scale);
        EXPECT_FLOAT_EQ(a.keypoints[i].orientation, b.keypoints[i].orientation);
        EXPECT_FLOAT_EQ(a.keypoints[i].response, b.keypoints[i].response);
        EXPECT_EQ(a.keypoints[i].octave, b.keypoints[i].octave);
        EXPECT_EQ(a.descriptors[i], b.descriptors[i]);
    }
}

TEST(ImageOrb, RgbInputMatchesGrayscale) {
    const Image gray = make_texture(96);
    const auto a = orb_detect_and_compute(gray, 200);
    const auto b = orb_detect_and_compute(gray2rgb(gray), 200);
    ASSERT_EQ(a.keypoints.size(), b.keypoints.size());
    for (std::size_t i = 0; i < a.keypoints.size(); ++i) {
        EXPECT_NEAR(a.keypoints[i].x, b.keypoints[i].x, 1e-5f);
        EXPECT_NEAR(a.keypoints[i].y, b.keypoints[i].y, 1e-5f);
    }
}

TEST(ImageOrb, DescriptorsSurviveNinetyDegreeRotation) {
    // imrotate90 is 90 degrees CCW with out(cols-1-c, r) = in(r, c), so a
    // feature at (x, y) moves to (y, 95 - x).
    const Image g = make_texture(96);
    const auto a = orb_detect_and_compute(g, 200);
    const auto b = orb_detect_and_compute(imrotate90(g), 200);
    ASSERT_FALSE(a.keypoints.empty());
    ASSERT_FALSE(b.keypoints.empty());
    EXPECT_LE(std::abs(static_cast<int>(b.keypoints.size()) - static_cast<int>(a.keypoints.size())), 10);

    const auto matches = match_descriptors(a.descriptors, b.descriptors, 0.8f, true);
    ASSERT_GE(matches.size(), 40u);
    int correct = 0;
    for (const auto& m : matches) {
        const auto& ka = a.keypoints[static_cast<std::size_t>(m.query_index)];
        const auto& kb = b.keypoints[static_cast<std::size_t>(m.train_index)];
        if (std::abs(kb.x - ka.y) <= 2.f && std::abs(kb.y - (95.f - ka.x)) <= 2.f) ++correct;
    }
    EXPECT_GE(static_cast<double>(correct) / static_cast<double>(matches.size()), 0.80);
}

TEST(ImageOrb, OrientationIsRotationCovariant) {
    const Image g = make_texture(96);
    const auto a = orb_detect_and_compute(g, 200);
    const auto b = orb_detect_and_compute(imrotate90(g), 200);
    const auto matches = match_descriptors(a.descriptors, b.descriptors, 0.8f, true);
    ASSERT_FALSE(matches.empty());
    double sum = 0.0;
    int n = 0, within = 0;
    for (const auto& m : matches) {
        const auto& ka = a.keypoints[static_cast<std::size_t>(m.query_index)];
        const auto& kb = b.keypoints[static_cast<std::size_t>(m.train_index)];
        if (!(std::abs(kb.x - ka.y) <= 2.f && std::abs(kb.y - (95.f - ka.x)) <= 2.f)) continue;
        double d = static_cast<double>(kb.orientation) - static_cast<double>(ka.orientation);
        while (d > M_PI) d -= 2.0 * M_PI;
        while (d <= -M_PI) d += 2.0 * M_PI;
        sum += d;
        ++n;
        if (std::abs(d + M_PI / 2.0) < 0.35) ++within;
    }
    ASSERT_GE(n, 20);
    // A 90-degree CCW image rotation maps (dx, dy) to (dy, -dx), i.e. it
    // subtracts pi/2 from every orientation in this module's convention.
    EXPECT_NEAR(-M_PI / 2.0, sum / static_cast<double>(n), 0.05);
    EXPECT_GE(static_cast<double>(within) / static_cast<double>(n), 0.80);
}

// ---- descriptor matching ----

TEST(ImageMatch, HammingDistanceBasics) {
    OrbDescriptor a{};
    OrbDescriptor b{};
    b[0] = static_cast<std::uint8_t>(0x0F);
    b[31] = static_cast<std::uint8_t>(0x80);
    OrbDescriptor ones{};
    ones.fill(static_cast<std::uint8_t>(0xFF));
    EXPECT_EQ(hamming_distance(a, a), 0);
    EXPECT_EQ(hamming_distance(a, b), 5);
    EXPECT_EQ(hamming_distance(b, a), 5);
    EXPECT_EQ(hamming_distance(a, ones), 256);
    EXPECT_EQ(hamming_distance(ones, ones), 0);
}

TEST(ImageMatch, L2DistanceBasics) {
    SiftDescriptor a{};
    SiftDescriptor b{};
    b[0] = 3.f;
    b[1] = 4.f;
    EXPECT_FLOAT_EQ(l2_distance(a, a), 0.f);
    EXPECT_FLOAT_EQ(l2_distance(a, b), 5.f);
    EXPECT_FLOAT_EQ(l2_distance(b, a), 5.f);
}

TEST(ImageMatch, RatioTestRejectsAmbiguousAndEmptyInputs) {
    OrbDescriptor a{};
    const std::vector<OrbDescriptor> query{a};
    const std::vector<OrbDescriptor> twins{a, a};
    const std::vector<OrbDescriptor> single{a};
    // Two identical train descriptors: d1 == d2 == 0, so 0 < 0.75*0 is false.
    EXPECT_TRUE(match_descriptors(query, twins, 0.75f, false).empty());
    // With only one train descriptor there is no second neighbour to compare
    // against, so the ratio test is skipped.
    const auto one = match_descriptors(query, single, 0.75f, false);
    ASSERT_EQ(one.size(), 1u);
    EXPECT_EQ(one[0].query_index, 0);
    EXPECT_EQ(one[0].train_index, 0);
    EXPECT_FLOAT_EQ(one[0].distance, 0.f);
    EXPECT_TRUE(match_descriptors(std::vector<OrbDescriptor>{}, single, 0.75f, false).empty());
    EXPECT_TRUE(match_descriptors(query, std::vector<OrbDescriptor>{}, 0.75f, false).empty());

    // The L2 overload behaves the same way.
    SiftDescriptor q{};
    SiftDescriptor near{};
    near[0] = 1.f;
    SiftDescriptor far{};
    far[0] = 10.f;
    SiftDescriptor mid{};
    mid[0] = 8.f;
    EXPECT_EQ(match_descriptors(std::vector<SiftDescriptor>{q},
                                std::vector<SiftDescriptor>{near, far}, 0.75f, false).size(), 1u);
    EXPECT_TRUE(match_descriptors(std::vector<SiftDescriptor>{q},
                                  std::vector<SiftDescriptor>{mid, far}, 0.75f, false).empty());
    // ratio_threshold <= 0 rejects everything; >= 1 disables the test.
    EXPECT_TRUE(match_descriptors(std::vector<SiftDescriptor>{q},
                                  std::vector<SiftDescriptor>{near, far}, 0.f, false).empty());
    EXPECT_EQ(match_descriptors(std::vector<SiftDescriptor>{q},
                                std::vector<SiftDescriptor>{mid, far}, 1.5f, false).size(), 1u);
}

TEST(ImageMatch, CrossCheckRemovesManyToOne) {
    OrbDescriptor zero{};
    OrbDescriptor one_bit{};
    one_bit[0] = static_cast<std::uint8_t>(0x01);
    const std::vector<OrbDescriptor> train{zero, one_bit};
    const std::vector<OrbDescriptor> query{zero, zero};

    const auto loose = match_descriptors(query, train, 0.75f, false);
    ASSERT_EQ(loose.size(), 2u);
    for (const auto& m : loose) {
        EXPECT_EQ(m.train_index, 0);
        EXPECT_FLOAT_EQ(m.distance, 0.f);
    }
    // Both queries want train 0; the reverse nearest neighbour is unique and
    // ties go to the smaller query index, so only query 0 survives.
    const auto strict = match_descriptors(query, train, 0.75f, true);
    ASSERT_EQ(strict.size(), 1u);
    EXPECT_EQ(strict[0].query_index, 0);
    EXPECT_EQ(strict[0].train_index, 0);
    EXPECT_FLOAT_EQ(strict[0].distance, 0.f);
}

TEST(ImageMatch, OrbSelfMatchIsIdentityWithZeroDistance) {
    const auto f = orb_detect_and_compute(make_texture(96), 200);
    ASSERT_FALSE(f.descriptors.empty());
    const auto m = match_descriptors(f.descriptors, f.descriptors, 0.8f, true);
    EXPECT_EQ(m.size(), f.descriptors.size());
    for (const auto& mm : m) {
        EXPECT_EQ(mm.query_index, mm.train_index);
        EXPECT_FLOAT_EQ(mm.distance, 0.f);
    }
}

// ---- SIFT ----

TEST(ImageSift, BlobKeypointsAreCenteredAndConsistent) {
    const auto f = sift_detect_and_compute(make_blob(64, 32, 32, 3.0));
    ASSERT_FALSE(f.keypoints.empty());
    ASSERT_EQ(f.keypoints.size(), f.descriptors.size());
    // An isotropic blob has no unique dominant orientation, so the 80%-of-peak
    // rule legitimately emits several keypoints at one point and scale.
    EXPECT_LE(f.keypoints.size(), 36u);
    const float resp0 = f.keypoints[0].response;
    for (const auto& k : f.keypoints) {
        EXPECT_NEAR(32.f, k.x, 0.5f);
        EXPECT_NEAR(32.f, k.y, 0.5f);
        EXPECT_EQ(k.octave, 0);
        EXPECT_GE(k.scale, 2.0f);
        EXPECT_LE(k.scale, 4.0f);
        EXPECT_NEAR(resp0, k.response, 1e-5f);
        EXPECT_GE(k.orientation, -static_cast<float>(M_PI) - 1e-5f);
        EXPECT_LE(k.orientation, static_cast<float>(M_PI) + 1e-5f);
    }
    // Every emitted orientation is distinct (duplicate suppression works).
    for (std::size_t i = 1; i < f.keypoints.size(); ++i)
        EXPECT_GT(std::abs(f.keypoints[i].orientation - f.keypoints[i - 1].orientation), 1e-3f);
}

TEST(ImageSift, ScaleSelectionTracksBlobSize) {
    const auto small = sift_detect_and_compute(make_blob(64, 32, 32, 3.0));
    const auto large = sift_detect_and_compute(make_blob(64, 32, 32, 6.0));
    ASSERT_FALSE(small.keypoints.empty());
    ASSERT_FALSE(large.keypoints.empty());
    EXPECT_GT(large.keypoints[0].scale, 1.5f * small.keypoints[0].scale);
    EXPECT_NEAR(32.f, large.keypoints[0].x, 0.5f);
    EXPECT_NEAR(32.f, large.keypoints[0].y, 0.5f);
    EXPECT_NEAR(32.f, small.keypoints[0].x, 0.5f);
    EXPECT_NEAR(32.f, small.keypoints[0].y, 0.5f);
}

TEST(ImageSift, TranslationCovariance) {
    const auto off = sift_detect_and_compute(make_blob(64, 40, 24, 3.0));
    ASSERT_FALSE(off.keypoints.empty());
    EXPECT_NEAR(40.f, off.keypoints[0].x, 0.5f);
    EXPECT_NEAR(24.f, off.keypoints[0].y, 0.5f);

    const Image field = make_blob_field(128);
    Image shifted(128, 128, 1, 0.5f);
    for (int r = 0; r + 8 < 128; ++r)
        for (int c = 0; c + 8 < 128; ++c) shifted.at(r + 8, c + 8, 0) = field.at(r, c, 0);
    const auto a = sift_detect_and_compute(field);
    const auto b = sift_detect_and_compute(shifted);
    ASSERT_FALSE(a.keypoints.empty());
    ASSERT_FALSE(b.keypoints.empty());
    int total = 0, reproduced = 0;
    for (const auto& k : a.keypoints) {
        if (k.x < 16.f || k.x > 104.f || k.y < 16.f || k.y > 104.f) continue;
        ++total;
        bool found = false;
        for (const auto& k2 : b.keypoints)
            if (std::abs(k2.x - (k.x + 8.f)) < 0.6f && std::abs(k2.y - (k.y + 8.f)) < 0.6f &&
                std::abs(k2.scale - k.scale) < 0.1f)
                found = true;
        if (found) ++reproduced;
    }
    ASSERT_GE(total, 20);
    EXPECT_GE(reproduced * 10, total * 9);
}

TEST(ImageSift, DegenerateInputsYieldNoKeypoints) {
    EXPECT_TRUE(sift_detect_and_compute(Image{}).keypoints.empty());
    EXPECT_TRUE(sift_detect_and_compute(Image(10, 10, 1, 0.5f)).keypoints.empty());
    EXPECT_TRUE(sift_detect_and_compute(Image(64, 64, 1, 0.5f)).keypoints.empty());
    // A step edge: the DoG is constant along the edge, so the 3D Hessian is
    // singular and every candidate is rejected by the linear solve.
    Image vstep(64, 64, 1, 0.f);
    for (int r = 0; r < 64; ++r)
        for (int c = 32; c < 64; ++c) vstep.at(r, c, 0) = 1.f;
    EXPECT_TRUE(sift_detect_and_compute(vstep).keypoints.empty());
    Image hstep(64, 64, 1, 0.f);
    for (int r = 32; r < 64; ++r)
        for (int c = 0; c < 64; ++c) hstep.at(r, c, 0) = 1.f;
    EXPECT_TRUE(sift_detect_and_compute(hstep).keypoints.empty());
    // Pixels live in [0, 1], so a contrast threshold of 3 is unreachable.
    EXPECT_TRUE(sift_detect_and_compute(make_blob(64, 32, 32, 3.0), 0, 3, 3.0f).keypoints.empty());
}

TEST(ImageSift, DescriptorsAreUnitNormAndNonNegative) {
    const auto f = sift_detect_and_compute(make_blob_field(128));
    ASSERT_EQ(f.keypoints.size(), f.descriptors.size());
    ASSERT_GE(f.descriptors.size(), 20u);
    float max_element = 0.f;
    for (const auto& d : f.descriptors) {
        double n = 0.0;
        for (const float v : d) {
            EXPECT_GE(v, 0.f);
            EXPECT_LE(v, 1.f);
            max_element = std::max(max_element, v);
            n += static_cast<double>(v) * static_cast<double>(v);
        }
        EXPECT_NEAR(1.0f, static_cast<float>(std::sqrt(n)), 1e-5f);
    }
    // Re-normalising after the 0.2 clip pushes the largest bins back above it.
    EXPECT_GT(max_element, 0.2f);
}

TEST(ImageSift, MaxFeaturesTruncatesAndSortsByResponse) {
    const Image img = make_blob_field(128);
    const auto all = sift_detect_and_compute(img);
    const auto few = sift_detect_and_compute(img, 3);
    ASSERT_GE(all.keypoints.size(), 3u);
    ASSERT_EQ(few.keypoints.size(), 3u);
    ASSERT_EQ(few.descriptors.size(), 3u);
    for (std::size_t i = 1; i < all.keypoints.size(); ++i)
        EXPECT_LE(all.keypoints[i].response, all.keypoints[i - 1].response);
    for (std::size_t i = 0; i < few.keypoints.size(); ++i) {
        EXPECT_FLOAT_EQ(few.keypoints[i].x, all.keypoints[i].x);
        EXPECT_FLOAT_EQ(few.keypoints[i].y, all.keypoints[i].y);
        EXPECT_FLOAT_EQ(few.keypoints[i].scale, all.keypoints[i].scale);
        EXPECT_FLOAT_EQ(few.keypoints[i].orientation, all.keypoints[i].orientation);
        EXPECT_FLOAT_EQ(few.keypoints[i].response, all.keypoints[i].response);
        EXPECT_EQ(few.keypoints[i].octave, all.keypoints[i].octave);
        EXPECT_EQ(few.descriptors[i], all.descriptors[i]);
    }
}

TEST(ImageSift, SelfMatchAndDeterminism) {
    const Image img = make_texture(96);
    const auto a = sift_detect_and_compute(img);
    const auto b = sift_detect_and_compute(img);
    ASSERT_EQ(a.keypoints.size(), b.keypoints.size());
    ASSERT_GE(a.keypoints.size(), 5u);
    for (std::size_t i = 0; i < a.keypoints.size(); ++i) {
        EXPECT_FLOAT_EQ(a.keypoints[i].x, b.keypoints[i].x);
        EXPECT_FLOAT_EQ(a.keypoints[i].y, b.keypoints[i].y);
        EXPECT_FLOAT_EQ(a.keypoints[i].scale, b.keypoints[i].scale);
        EXPECT_FLOAT_EQ(a.keypoints[i].orientation, b.keypoints[i].orientation);
        EXPECT_EQ(a.descriptors[i], b.descriptors[i]);
        EXPECT_FLOAT_EQ(l2_distance(a.descriptors[i], b.descriptors[i]), 0.f);
    }
    const auto m = match_descriptors(a.descriptors, a.descriptors, 0.8f, true);
    EXPECT_EQ(m.size(), a.descriptors.size());
    for (const auto& mm : m) {
        EXPECT_EQ(mm.query_index, mm.train_index);
        EXPECT_NEAR(0.f, mm.distance, 1e-6f);
    }
}

// ---------------------------------------------------------------------------
// §8.4: the detectors' output, pinned.
//
// A sample of 24 mutants aimed at the detector and pyramid code scored 52.2%,
// and the survivors said why in one line: `ImageOrb.IsDeterministic` compares
// ORB against ANOTHER CALL OF ORB. That is the same blindness a codec's
// round-trip has -- any change applied consistently is invisible to it -- and
// for a detector, "consistently" covers the whole pyramid: the number of
// levels, each level's dimensions, the per-level feature budget, and the
// octave a keypoint is attributed to. Every existing assertion here is a bound
// (`0 <= x <= 95`, responses descending, orientation in [-pi, pi]) or a
// comparison between two outputs, and a pyramid one level taller satisfies all
// of them.
//
// These pin the output itself. The float fields are compared at 1e-3 rather
// than exactly: they are products of an integer coordinate with a power of the
// scale factor, so they agree far more closely than that between compilers,
// while a structural change moves them by whole pixels. The counts and octaves
// are exact, because those are integers and a pyramid of a different height
// changes them outright.
//
// Captured from the implementation, so this is a regression guard rather than
// a proof -- what makes it worth having is the property tests around it. The
// covariance, scale-selection, unit-norm and sortedness tests above establish
// that the answers are RIGHT; this freezes the freedom they leave.
// ---------------------------------------------------------------------------

TEST(ImageFeatureGolden, OrbOnTheTextureIsExactlyThisPyramid) {
    const auto f = orb_detect_and_compute(make_texture(96), 200);
    ASSERT_EQ(f.keypoints.size(), f.descriptors.size());

    // 176 rather than the 200 asked for: the coarse levels run out of corners
    // before their budget does. A pyramid with one more or one fewer level
    // reaches a different number.
    EXPECT_EQ(f.keypoints.size(), 176u);

    int octmax = 0;
    std::vector<int> per_octave(16, 0);
    for (const auto& k : f.keypoints) {
        ASSERT_GE(k.octave, 0);
        ASSERT_LT(k.octave, 16);
        octmax = std::max(octmax, k.octave);
        ++per_octave[static_cast<std::size_t>(k.octave)];
    }
    EXPECT_EQ(octmax, 5) << "the pyramid is not the height it was";
    // 96 / 1.2^k stops being wide enough for a FAST ring at k = 6, so six
    // levels is what this image supports and every one of them contributes.
    for (int i = 0; i <= 5; ++i) {
        EXPECT_GT(per_octave[static_cast<std::size_t>(i)], 0) << "level " << i << " empty";
    }

    struct Want { float x, y, scale, orientation, response; int octave; };
    const Want want[12] = {
        {51.8400f, 36.0000f, 1.4400f,  3.1393f, 5.98135f, 2},
        {52.0000f, 36.0000f, 1.0000f,  0.0730f, 5.84092f, 0},
        {51.6000f, 36.0000f, 1.2000f, -1.0642f, 5.79097f, 1},
        {31.6800f, 27.3600f, 1.4400f,  2.3822f, 5.72686f, 2},
        {36.0000f, 24.0000f, 1.2000f,  2.1742f, 5.70816f, 1},
        {36.0000f, 24.0000f, 1.0000f,  1.6318f, 5.68782f, 0},
        {51.8400f, 36.2880f, 1.7280f, -2.1571f, 5.56609f, 3},
        {73.0000f, 76.0000f, 1.0000f, -1.9233f, 5.20680f, 0},
        {31.1040f, 27.6480f, 1.7280f,  2.1588f, 5.16292f, 3},
        {31.2000f, 26.4000f, 1.2000f,  2.0368f, 5.10253f, 1},
        {39.6000f, 27.6000f, 1.2000f,  3.0478f, 4.98170f, 1},
        {73.2000f, 75.6000f, 1.2000f, -1.4108f, 4.96723f, 1},
    };
    ASSERT_GE(f.keypoints.size(), 12u);
    for (int i = 0; i < 12; ++i) {
        const auto& k = f.keypoints[static_cast<std::size_t>(i)];
        const Want& w = want[i];
        EXPECT_NEAR(k.x, w.x, 1e-3f) << "keypoint " << i << " column";
        EXPECT_NEAR(k.y, w.y, 1e-3f) << "keypoint " << i << " row";
        EXPECT_NEAR(k.scale, w.scale, 1e-3f) << "keypoint " << i << " scale";
        EXPECT_NEAR(k.orientation, w.orientation, 1e-3f) << "keypoint " << i << " orientation";
        EXPECT_NEAR(k.response, w.response, 1e-3f) << "keypoint " << i << " response";
        EXPECT_EQ(k.octave, w.octave) << "keypoint " << i << " octave";
    }

    // A budget small enough to bind. At 200 the levels run out of corners first
    // -- 176 come back -- so the per-level allocation is never the constraint and
    // an allocation off by one is invisible. At 24 it IS the constraint, and the
    // count is then exactly what the geometric share plus the last level's
    // remainder come to.
    const auto tight = orb_detect_and_compute(make_texture(96), 24);
    EXPECT_EQ(tight.keypoints.size(), 24u) << "the per-level budget did not sum to what was asked";
    EXPECT_EQ(tight.descriptors.size(), tight.keypoints.size());
    const auto tight7 = orb_detect_and_compute(make_texture(96), 7);
    EXPECT_EQ(tight7.keypoints.size(), 7u);

    // The same keypoint found at three scales -- (51.84, 36) at levels 2, 0 and
    // 1 in the first three rows -- is the pyramid working, and the exact
    // coordinates are the level-to-base mapping (r * s, c * s) the comment in
    // `orb_detect_and_compute` promises. A level sized one row wider breaks it.
    EXPECT_NEAR(f.keypoints[0].x / f.keypoints[0].scale, 36.0f, 1e-3f);
    EXPECT_NEAR(f.keypoints[2].x / f.keypoints[2].scale, 43.0f, 1e-3f);
}

TEST(ImageFeatureGolden, SiftOnTheBlobFieldIsExactlyThisScaleSpace) {
    const auto f = sift_detect_and_compute(make_blob_field(128), 200);
    ASSERT_EQ(f.keypoints.size(), f.descriptors.size());
    EXPECT_EQ(f.keypoints.size(), 160u);

    int lo = 99, hi = -99;
    for (const auto& k : f.keypoints) { lo = std::min(lo, k.octave); hi = std::max(hi, k.octave); }
    // floor(log2(128)) - 2 = 5 octaves are built; the blobs (sigma 2.5 to 5)
    // are resolved in the first two, and nothing survives past them. An octave
    // count computed as floor(log2(n)) + 2 instead searches four more.
    EXPECT_EQ(lo, 0);
    EXPECT_EQ(hi, 1) << "the scale space is not the depth it was";

    struct Want { float x, y, scale, orientation, response; int octave; };
    const Want want[8] = {
        {112.0000f, 48.0000f, 2.1927f, -2.2573f, 0.059632f, 0},
        {112.0000f, 48.0000f, 2.1927f, -1.5708f, 0.059632f, 0},
        {112.0000f, 48.0000f, 2.1927f, -0.6080f, 0.059632f, 0},
        {112.0000f, 48.0000f, 2.1927f, -0.0000f, 0.059632f, 0},
        {112.0000f, 48.0000f, 2.1927f,  0.7373f, 0.059632f, 0},
        {112.0000f, 48.0000f, 2.1927f,  1.5708f, 0.059632f, 0},
        {112.0000f, 48.0000f, 2.1927f,  2.2335f, 0.059632f, 0},
        {112.0000f, 48.0000f, 2.1927f,  3.1416f, 0.059632f, 0},
    };
    ASSERT_GE(f.keypoints.size(), 8u);
    for (int i = 0; i < 8; ++i) {
        const auto& k = f.keypoints[static_cast<std::size_t>(i)];
        const Want& w = want[i];
        EXPECT_NEAR(k.x, w.x, 1e-3f) << "keypoint " << i << " column";
        EXPECT_NEAR(k.y, w.y, 1e-3f) << "keypoint " << i << " row";
        EXPECT_NEAR(k.scale, w.scale, 1e-3f) << "keypoint " << i << " scale";
        EXPECT_NEAR(k.orientation, w.orientation, 1e-3f) << "keypoint " << i << " orientation";
        EXPECT_NEAR(k.response, w.response, 1e-4f) << "keypoint " << i << " response";
        EXPECT_EQ(k.octave, w.octave) << "keypoint " << i << " octave";
    }

    // Eight keypoints at one location, one per orientation histogram peak: a
    // blob on a 4x4 grid of blobs is locally symmetric, so the peaks come in
    // at the grid's own angles. That is the orientation assignment's output and
    // nothing else asserted it.
    for (int i = 1; i < 8; ++i) {
        EXPECT_NEAR(f.keypoints[static_cast<std::size_t>(i)].x, f.keypoints[0].x, 1e-3f);
        EXPECT_NEAR(f.keypoints[static_cast<std::size_t>(i)].y, f.keypoints[0].y, 1e-3f);
        EXPECT_GT(f.keypoints[static_cast<std::size_t>(i)].orientation,
                  f.keypoints[static_cast<std::size_t>(i - 1)].orientation);
    }

    // The first descriptor's leading two spatial rows. The 4x4x8 layout means
    // these are bins (0,0,*) and (0,1,*), and the zeros are as much of the
    // contract as the peaks: a blob's gradient points one way in each quadrant,
    // so most orientation bins in a cell are empty. Descriptor binning that
    // wrote to the neighbouring bin would fill them.
    const float want_d[16] = {
        0.00172f, 0.01530f, 0.00130f, 0.00000f, 0.00000f, 0.00000f, 0.00000f, 0.00000f,
        0.00121f, 0.07209f, 0.09400f, 0.00814f, 0.00001f, 0.00000f, 0.00000f, 0.00000f,
    };
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(f.descriptors[0][static_cast<std::size_t>(i)], want_d[i], 2e-4f)
            << "descriptor component " << i;
    }
}
