// Regression tests for three audited defects.
//
//  * lz77_encode wrote a filler next_char of 0 when a match ran to the end of
//    the input, so the value 0 meant both "the byte 0x00" and "no literal";
//    lz77_decode resolved that by dropping the byte whenever length > 0, making
//    the round trip LOSSY for any input containing 0x00 after a match.
//  * sample_entropy counted its length-m and length-(m+1) template matches over
//    populations of different sizes (n-m and n-m-1), so A/B < 1 even when every
//    pair matched and a perfectly regular series reported positive entropy.
//  * lz_complexity forbade the copy from overlapping the position being
//    reproduced, which is precisely what lets a periodic tail be reproduced by a
//    single production, so periodic sequences scored as though they were random.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "ms/compress/compress.hpp"
#include "ms/info/info.hpp"

namespace {

bool round_trips(const std::vector<std::uint8_t>& data) {
    const auto tokens = ms::compress::lz77_encode(data, 32, 16);
    return ms::compress::lz77_decode(tokens) == data;
}

} // namespace

TEST(Lz77RoundTrip, PreservesALiteralZeroAfterAMatch) {
    // The reported case: {65, 66, 66, 0, 66, 66} decoded to {65, 66, 66, 65, 66},
    // one byte short with the 0x00 gone.
    const std::vector<std::uint8_t> data{65, 66, 66, 0, 66, 66};
    const auto tokens = ms::compress::lz77_encode(data, 32, 16);
    const auto back = ms::compress::lz77_decode(tokens);
    ASSERT_EQ(back.size(), data.size());
    EXPECT_EQ(back, data);
}

TEST(Lz77RoundTrip, IsLosslessOnZeroHeavyInput) {
    EXPECT_TRUE(round_trips({0}));
    EXPECT_TRUE(round_trips({0, 0, 0, 0, 0, 0, 0, 0}));
    EXPECT_TRUE(round_trips({1, 2, 3, 0, 1, 2, 3, 0, 9}));
    EXPECT_TRUE(round_trips({0, 1, 0, 1, 0, 1, 0}));
    EXPECT_TRUE(round_trips({}));
}

TEST(Lz77RoundTrip, IsLosslessOnRepetitiveAndRandomInput) {
    EXPECT_TRUE(round_trips({7, 7, 7, 7, 7, 7, 7, 7, 7, 7}));

    std::vector<std::uint8_t> pseudo(512);
    std::uint32_t s = 2463534242u;
    for (auto& b : pseudo) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        b = static_cast<std::uint8_t>(s & 0xFFu);
    }
    EXPECT_TRUE(round_trips(pseudo));

    // Every byte value, so 0x00 appears in many positions relative to matches.
    std::vector<std::uint8_t> all(256);
    for (std::size_t i = 0; i < all.size(); ++i) {
        all[i] = static_cast<std::uint8_t>(i);
    }
    EXPECT_TRUE(round_trips(all));
    std::vector<std::uint8_t> twice(all);
    twice.insert(twice.end(), all.begin(), all.end());
    EXPECT_TRUE(round_trips(twice));
}

TEST(SampleEntropy, IsZeroForAPerfectlyRegularSeries) {
    // Every template matches every other, so A == B and SampEn == 0. The
    // mismatched populations used to make this strictly positive.
    const std::vector<double> constant(200, 1.0);
    EXPECT_NEAR(ms::info::sample_entropy(constant, 2, 0.2), 0.0, 1e-12);

    std::vector<double> periodic(200);
    for (std::size_t i = 0; i < periodic.size(); ++i) {
        periodic[i] = (i % 2 == 0) ? 0.0 : 1.0;
    }
    EXPECT_NEAR(ms::info::sample_entropy(periodic, 2, 0.2), 0.0, 1e-12);
}

TEST(SampleEntropy, IsLargerForAnIrregularSeries) {
    std::vector<double> noisy(400);
    std::uint32_t s = 88675123u;
    for (auto& v : noisy) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        v = static_cast<double>(s % 1000u) / 1000.0;
    }
    const std::vector<double> constant(400, 1.0);
    EXPECT_GT(ms::info::sample_entropy(noisy, 2, 0.2),
              ms::info::sample_entropy(constant, 2, 0.2));
}

TEST(SampleEntropy, DegenerateInputIsHandled) {
    const std::vector<double> tiny{1.0, 2.0};
    EXPECT_DOUBLE_EQ(ms::info::sample_entropy(tiny, 5, 0.2), 0.0);  // n < m+1
    EXPECT_DOUBLE_EQ(ms::info::sample_entropy({}, 2, 0.2), 0.0);
    const std::vector<double> spread{0.0, 100.0, 200.0, 300.0, 400.0};
    EXPECT_TRUE(std::isfinite(ms::info::sample_entropy(spread, 2, 0.001)));  // no matches
}

TEST(LzComplexity, PeriodicSequencesScoreFarBelowRandom) {
    const std::size_t n = 1000;
    std::vector<int> periodic(n);
    for (std::size_t i = 0; i < n; ++i) {
        periodic[i] = static_cast<int>(i % 2);
    }
    const std::vector<int> constant(n, 1);

    std::vector<int> random_bits(n);
    std::uint32_t s = 12345u;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1103515245u + 12345u;
        random_bits[i] = static_cast<int>((s >> 16) & 1u);
    }

    const double c_const = ms::info::lz_complexity(constant);
    const double c_per = ms::info::lz_complexity(periodic);
    const double c_rnd = ms::info::lz_complexity(random_bits);

    // A random binary sequence has normalised LZ complexity near 1; a periodic
    // one is a small multiple of the constant case. Before the fix the copy
    // could not overlap, so the periodic sequence was charged a fresh phrase per
    // period and scored like noise.
    EXPECT_GT(c_rnd, 0.5);
    EXPECT_LT(c_per, 0.1);
    EXPECT_LT(c_const, c_per);
    EXPECT_GT(c_rnd, 10.0 * c_per);
}

TEST(LzComplexity, DegenerateInputIsHandled) {
    EXPECT_DOUBLE_EQ(ms::info::lz_complexity({}), 0.0);
    const std::vector<int> one{1};
    EXPECT_GT(ms::info::lz_complexity(one), 0.0);
    const std::vector<int> two{1, 1};
    EXPECT_TRUE(std::isfinite(ms::info::lz_complexity(two)));
}
