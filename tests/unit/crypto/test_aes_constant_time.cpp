// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// The AES S-box after it stopped being an array index.
//
// kAesSbox[state[i]] and kAesInvSbox[state[i]] were the only secret-dependent memory
// accesses in the cipher, and the index is derived from the key. Which of the four
// cache lines each table spans gets touched leaks that index, which is the classic
// AES cache-timing channel. Both reads now go through a masked scan of all 256
// entries, so the address sequence no longer depends on the value.
//
// A masked scan is easy to get subtly wrong -- an off-by-one in the mask picks the
// neighbouring entry, and a sign error picks none at all and returns zero. These
// tests exercise every index of both tables and pin the result against the published
// vectors, so a wrong mask cannot pass.

#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "ms/crypto/crypto.hpp"

namespace {

std::vector<std::uint8_t> from_hex(const std::string& hex) {
    std::vector<std::uint8_t> out;
    out.reserve(hex.size() / 2);
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        out.push_back(static_cast<std::uint8_t>(std::stoul(hex.substr(i, 2), nullptr, 16)));
    }
    return out;
}

std::string to_hex(const std::vector<std::uint8_t>& bytes) {
    static const char* digits = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (const std::uint8_t b : bytes) {
        out.push_back(digits[b >> 4]);
        out.push_back(digits[b & 0x0F]);
    }
    return out;
}

} // namespace

TEST(AesConstantTime, EveryForwardSboxIndexRoundTrips) {
    // Sweeping the first plaintext byte through 0..255 drives every index of the
    // forward table through the first SubBytes, and the round trip drives every index
    // of the inverse table. A mask that selects the wrong entry, or none, breaks here
    // for whichever index it mishandles.
    const auto key = from_hex("2b7e151628aed2a6abf7158809cf4f3c");
    for (int first = 0; first < 256; ++first) {
        std::vector<std::uint8_t> block = from_hex("003243f6a8885a308d313198a2e03707");
        block[0] = static_cast<std::uint8_t>(first);
        const auto cipher = ms::crypto::aes128_encrypt_block(key, block);
        ASSERT_EQ(cipher.size(), 16U) << "index " << first;
        const auto plain = ms::crypto::aes128_decrypt_block(key, cipher);
        EXPECT_EQ(plain, block) << "round trip failed at first byte " << first;
    }
}

TEST(AesConstantTime, EveryKeyByteIndexRoundTrips) {
    // The key expansion reads kAesSbox[word[i]] with the index taken straight from key
    // material, which is the most sensitive of the three sites. Sweeping a key byte
    // covers every index there.
    const auto block = from_hex("6bc1bee22e409f96e93d7e117393172a");
    for (int b = 0; b < 256; ++b) {
        auto key = from_hex("2b7e151628aed2a6abf7158809cf4f3c");
        key[15] = static_cast<std::uint8_t>(b);
        const auto cipher = ms::crypto::aes128_encrypt_block(key, block);
        ASSERT_EQ(cipher.size(), 16U) << "key byte " << b;
        const auto plain = ms::crypto::aes128_decrypt_block(key, cipher);
        EXPECT_EQ(plain, block) << "round trip failed at key byte " << b;
    }
}

TEST(AesConstantTime, PublishedVectorsAreUnchanged) {
    // Round-tripping alone would still pass if both tables were wrong in mirrored
    // ways. These are the FIPS-197 / SP 800-38A vectors, which pin the absolute
    // values.
    struct Case {
        const char* key;
        const char* plaintext;
        const char* ciphertext;
        bool aes256;
    };
    const Case cases[] = {
        // FIPS-197 C.1 (AES-128) and C.3 (AES-256).
        {"000102030405060708090a0b0c0d0e0f", "00112233445566778899aabbccddeeff",
         "69c4e0d86a7b0430d8cdb78070b4c55a", false},
        {"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
         "00112233445566778899aabbccddeeff", "8ea2b7ca516745bfeafc49904b496089", true},
        // SP 800-38A F.1.1 ECB-AES128, first two blocks.
        {"2b7e151628aed2a6abf7158809cf4f3c", "6bc1bee22e409f96e93d7e117393172a",
         "3ad77bb40d7a3660a89ecaf32466ef97", false},
        {"2b7e151628aed2a6abf7158809cf4f3c", "ae2d8a571e03ac9c9eb76fac45af8e51",
         "f5d3d58503b9699de785895a96fdbaaf", false},
        // SP 800-38A F.1.5 ECB-AES256, first block.
        {"603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4",
         "6bc1bee22e409f96e93d7e117393172a", "f3eed1bdb5d2a03c064b5a7e3db181f8", true},
    };
    for (const Case& c : cases) {
        const auto key = from_hex(c.key);
        const auto plaintext = from_hex(c.plaintext);
        const auto cipher = c.aes256 ? ms::crypto::aes256_encrypt_block(key, plaintext)
                                     : ms::crypto::aes128_encrypt_block(key, plaintext);
        EXPECT_EQ(to_hex(cipher), std::string(c.ciphertext)) << "encrypting " << c.plaintext;
        const auto back = c.aes256 ? ms::crypto::aes256_decrypt_block(key, cipher)
                                   : ms::crypto::aes128_decrypt_block(key, cipher);
        EXPECT_EQ(to_hex(back), std::string(c.plaintext)) << "decrypting " << c.ciphertext;
    }
}

TEST(AesConstantTime, MaskSelectsExactlyOneEntry) {
    // The mask arithmetic on its own, stated where it can be read. aes_table_lookup_ct
    // is file-local, so this reproduces its expression rather than calling it: for the
    // matching index the mask must be all ones and for every other index all zeros. An
    // off-by-one here is the difference between reading the S-box and reading the
    // entry beside it.
    for (int index = 0; index < 256; ++index) {
        int selected = 0;
        for (int i = 0; i < 256; ++i) {
            const auto diff = static_cast<std::uint8_t>(static_cast<std::uint8_t>(i) ^
                                                        static_cast<std::uint8_t>(index));
            const auto mask = static_cast<std::uint8_t>((static_cast<int>(diff) - 1) >> 8);
            if (i == index) {
                EXPECT_EQ(mask, 0xFF) << "index " << index << " did not select itself";
                ++selected;
            } else {
                EXPECT_EQ(mask, 0x00) << "index " << index << " also selected " << i;
            }
        }
        EXPECT_EQ(selected, 1);
    }
}
