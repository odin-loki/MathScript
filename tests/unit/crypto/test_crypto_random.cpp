// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "ms/crypto/crypto.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <span>
#include <vector>

#include <gtest/gtest.h>

using namespace ms::crypto;

// random_bytes now draws from the operating system CSPRNG (getrandom/BCryptGenRandom/
// arc4random_buf//dev/urandom) instead of std::random_device, and fails closed by
// returning an empty vector. These tests pin the contract and the statistical sanity of
// whichever backend the platform selected; every bound below is chosen many sigma wide
// so that a healthy generator cannot flake.

// A zero-length request yields an empty buffer and never touches the OS.
TEST(CryptoRandomBytesOs, ZeroLengthIsEmpty) {
    const auto out = random_bytes(0);
    EXPECT_TRUE(out.empty());
}

// Every request returns exactly the requested number of bytes. A short or empty result
// is this module's failure signal, so this also asserts that the OS CSPRNG path
// succeeded on this platform rather than silently falling all the way through.
TEST(CryptoRandomBytesOs, ExactRequestedLength) {
    for (std::size_t n : {std::size_t{1}, std::size_t{15}, std::size_t{16},
                          std::size_t{32}, std::size_t{257}, std::size_t{4096}}) {
        const auto out = random_bytes(n);
        EXPECT_EQ(out.size(), n) << "n=" << n;
    }
}

// Independent calls must not repeat. 64 draws of 32 bytes give 64 distinct values; a
// collision has probability ~2^-249 even for a broken-but-advancing generator.
TEST(CryptoRandomBytesOs, IndependentCallsDiffer) {
    const auto a = random_bytes(32);
    const auto b = random_bytes(32);
    ASSERT_EQ(a.size(), 32u);
    ASSERT_EQ(b.size(), 32u);
    EXPECT_NE(a, b);

    std::set<std::vector<std::uint8_t>> seen;
    for (int i = 0; i < 64; ++i) seen.insert(random_bytes(32));
    EXPECT_EQ(seen.size(), 64u);
}

// A large single request is filled completely -- this is what exercises the chunking and
// short-read loops -- and is not a constant buffer.
TEST(CryptoRandomBytesOs, LargeRequestIsFilled) {
    constexpr std::size_t kN = std::size_t{1} << 20;  // 1 MiB
    const auto out = random_bytes(kN);
    ASSERT_EQ(out.size(), kN);

    bool all_zero = true;
    bool all_same = true;
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (out[i] != 0) all_zero = false;
        if (out[i] != out[0]) all_same = false;
        if (!all_zero && !all_same) break;
    }
    EXPECT_FALSE(all_zero);
    EXPECT_FALSE(all_same);
}

// Rough uniformity: 2^20 bytes over 256 buckets, expected 4096 per bucket. Pearson
// chi-square with 255 degrees of freedom has mean 255 and standard deviation
// sqrt(2*255) ~= 22.6, so the 500 bound is ~10.8 sigma out and the false-failure rate is
// far below 1e-20. A healthy CSPRNG lands around 230-290 (measured here: 247).
TEST(CryptoRandomBytesOs, ByteValuesAreRoughlyUniform) {
    constexpr std::size_t kN = std::size_t{1} << 20;
    const auto out = random_bytes(kN);
    ASSERT_EQ(out.size(), kN);

    std::array<std::size_t, 256> hist{};
    for (std::uint8_t v : out) hist[static_cast<std::size_t>(v)]++;

    const double expected = static_cast<double>(kN) / 256.0;
    double chi_square = 0.0;
    for (std::size_t bucket : hist) {
        // No bucket may be wildly off on its own either (a ~32 sigma window).
        EXPECT_GT(static_cast<double>(bucket), expected * 0.5);
        EXPECT_LT(static_cast<double>(bucket), expected * 1.5);
        const double d = static_cast<double>(bucket) - expected;
        chi_square += d * d / expected;
    }
    EXPECT_LT(chi_square, 500.0) << "chi-square " << chi_square << " over 255 dof";
    // Every one of the 256 byte values must occur at least once.
    for (std::size_t bucket : hist) EXPECT_GT(bucket, 0u);
}

// The span overload fills the caller's buffer and reports success explicitly.
TEST(CryptoRandomBytesOs, RandomBytesIntoFillsCallerBuffer) {
    std::vector<std::uint8_t> buf(64, 0u);
    EXPECT_TRUE(random_bytes_into(std::span<std::uint8_t>(buf.data(), buf.size())));
    bool all_zero = true;
    for (std::uint8_t v : buf) {
        if (v != 0) { all_zero = false; break; }
    }
    EXPECT_FALSE(all_zero);

    // Filling an empty span is a trivial success and performs no syscall.
    std::vector<std::uint8_t> empty_buf;
    EXPECT_TRUE(random_bytes_into(std::span<std::uint8_t>(empty_buf.data(), empty_buf.size())));
}

// Two independent fills of the same buffer must differ, and a fill must overwrite every
// byte rather than leaving a tail of the previous contents.
TEST(CryptoRandomBytesOs, RandomBytesIntoOverwritesWholeBuffer) {
    std::vector<std::uint8_t> first(128, 0u);
    std::vector<std::uint8_t> second(128, 0u);
    ASSERT_TRUE(random_bytes_into(std::span<std::uint8_t>(first.data(), first.size())));
    ASSERT_TRUE(random_bytes_into(std::span<std::uint8_t>(second.data(), second.size())));
    EXPECT_NE(first, second);

    std::vector<std::uint8_t> sentinel(4096, 0xAB);
    ASSERT_TRUE(random_bytes_into(std::span<std::uint8_t>(sentinel.data(), sentinel.size())));
    std::size_t untouched_tail = 0;
    for (std::size_t i = sentinel.size(); i-- > 0;) {
        if (sentinel[i] != 0xAB) break;
        ++untouched_tail;
    }
    // A run of 0xAB longer than 8 bytes at the end has probability 256^-9.
    EXPECT_LT(untouched_tail, 8u);
}

// Adjacent 8-byte blocks inside one draw must not repeat: this catches a backend that
// fills only the first chunk and leaves the rest duplicated or untouched.
TEST(CryptoRandomBytesOs, NoRepeatedBlocksWithinOneDraw) {
    const auto out = random_bytes(8192);
    ASSERT_EQ(out.size(), 8192u);
    std::set<std::vector<std::uint8_t>> blocks;
    for (std::size_t i = 0; i + 8 <= out.size(); i += 8)
        blocks.insert(std::vector<std::uint8_t>(out.begin() + static_cast<std::ptrdiff_t>(i),
                                                out.begin() + static_cast<std::ptrdiff_t>(i + 8)));
    EXPECT_EQ(blocks.size(), out.size() / 8);
}

// Bit balance: over 2^20 bytes the number of set bits must sit near 50%.
// Binomial(2^23, 0.5) has standard deviation ~1448 bits, so the +/-1% window is ~58 sigma.
TEST(CryptoRandomBytesOs, BitBalanceIsNearHalf) {
    constexpr std::size_t kN = std::size_t{1} << 20;
    const auto out = random_bytes(kN);
    ASSERT_EQ(out.size(), kN);
    std::size_t ones = 0;
    for (std::uint8_t v : out)
        for (int b = 0; b < 8; ++b) ones += static_cast<std::size_t>((v >> b) & 1u);
    const double frac = static_cast<double>(ones) / static_cast<double>(kN * 8);
    EXPECT_GT(frac, 0.49);
    EXPECT_LT(frac, 0.51);
}
