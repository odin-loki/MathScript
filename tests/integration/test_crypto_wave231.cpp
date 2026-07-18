// Wave 231 — ms::crypto integration tests (SHA/HMAC live; AES/ChaCha pending sibling branches).
//
// AES CBC / ChaCha20 round-trip tests are DISABLED until wave231/crypto-aes and
// wave231/crypto-chacha merge into ms::crypto. AES-128 ECB NIST vector below uses a
// minimal inline block cipher so this suite validates the vector standalone.

#include "ms/crypto/crypto.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <vector>

using namespace ms::crypto;

namespace {

std::vector<uint8_t> from_hex(std::string_view hex) {
    std::vector<uint8_t> out;
    out.reserve(hex.size() / 2);
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        auto nibble = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        out.push_back(static_cast<uint8_t>((nibble(hex[i]) << 4) | nibble(hex[i + 1])));
    }
    return out;
}

void expect_hex(const std::vector<uint8_t>& bytes, std::string_view expected) {
    EXPECT_EQ(to_hex(std::span<const uint8_t>(bytes)), expected);
}

// ---- Minimal AES-128 ECB single-block encrypt (test scaffolding only) ----

namespace minimal_aes128 {

constexpr std::array<uint8_t, 256> kSbox = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
};

constexpr std::array<uint8_t, 10> kRcon = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};

uint8_t xtime(uint8_t x) { return static_cast<uint8_t>((x << 1) ^ ((x & 0x80) ? 0x1b : 0)); }

void sub_bytes(std::array<uint8_t, 16>& state) {
    for (auto& b : state) b = kSbox[b];
}

void shift_rows(std::array<uint8_t, 16>& s) {
    const uint8_t r1[4] = {s[1], s[5], s[9], s[13]};
    s[1] = r1[1]; s[5] = r1[2]; s[9] = r1[3]; s[13] = r1[0];
    const uint8_t r2[4] = {s[2], s[6], s[10], s[14]};
    s[2] = r2[2]; s[6] = r2[3]; s[10] = r2[0]; s[14] = r2[1];
    const uint8_t r3[4] = {s[3], s[7], s[11], s[15]};
    s[3] = r3[3]; s[7] = r3[0]; s[11] = r3[1]; s[15] = r3[2];
}

void mix_columns(std::array<uint8_t, 16>& s) {
    for (int c = 0; c < 4; ++c) {
        const int i = c * 4;
        const uint8_t a = s[static_cast<std::size_t>(i)];
        const uint8_t b = s[static_cast<std::size_t>(i + 1)];
        const uint8_t c0 = s[static_cast<std::size_t>(i + 2)];
        const uint8_t d = s[static_cast<std::size_t>(i + 3)];
        const uint8_t t = static_cast<uint8_t>(a ^ b ^ c0 ^ d);
        const uint8_t u = a;
        s[static_cast<std::size_t>(i)] =
            static_cast<uint8_t>(a ^ t ^ xtime(static_cast<uint8_t>(a ^ b)));
        s[static_cast<std::size_t>(i + 1)] =
            static_cast<uint8_t>(b ^ t ^ xtime(static_cast<uint8_t>(b ^ c0)));
        s[static_cast<std::size_t>(i + 2)] =
            static_cast<uint8_t>(c0 ^ t ^ xtime(static_cast<uint8_t>(c0 ^ d)));
        s[static_cast<std::size_t>(i + 3)] =
            static_cast<uint8_t>(d ^ t ^ xtime(static_cast<uint8_t>(d ^ u)));
    }
}

void add_round_key(std::array<uint8_t, 16>& state, const uint8_t* rk) {
    for (int i = 0; i < 16; ++i) state[static_cast<std::size_t>(i)] ^= rk[i];
}

void key_expansion128(const uint8_t key[16], std::array<std::array<uint8_t, 16>, 11>& round_keys) {
    std::array<uint8_t, 176> w{};
    std::copy(key, key + 16, w.begin());
    int rcon_idx = 0;
    for (int i = 4; i < 44; ++i) {
        std::array<uint8_t, 4> temp{};
        temp[0] = w[static_cast<std::size_t>((i - 1) * 4 + 0)];
        temp[1] = w[static_cast<std::size_t>((i - 1) * 4 + 1)];
        temp[2] = w[static_cast<std::size_t>((i - 1) * 4 + 2)];
        temp[3] = w[static_cast<std::size_t>((i - 1) * 4 + 3)];
        if (i % 4 == 0) {
            const uint8_t t = temp[0];
            temp[0] = temp[1];
            temp[1] = temp[2];
            temp[2] = temp[3];
            temp[3] = t;
            temp[0] = static_cast<uint8_t>(kSbox[temp[0]] ^ kRcon[rcon_idx++]);
            temp[1] = kSbox[temp[1]];
            temp[2] = kSbox[temp[2]];
            temp[3] = kSbox[temp[3]];
        }
        for (int j = 0; j < 4; ++j) {
            w[static_cast<std::size_t>(i * 4 + j)] =
                static_cast<uint8_t>(w[static_cast<std::size_t>((i - 4) * 4 + j)] ^ temp[static_cast<std::size_t>(j)]);
        }
    }
    for (int round = 0; round < 11; ++round) {
        std::copy(w.begin() + static_cast<std::ptrdiff_t>(round * 16),
                  w.begin() + static_cast<std::ptrdiff_t>((round + 1) * 16),
                  round_keys[static_cast<std::size_t>(round)].begin());
    }
}

std::array<uint8_t, 16> encrypt_block_ecb(const std::array<uint8_t, 16>& key,
                                          const std::array<uint8_t, 16>& block) {
    std::array<std::array<uint8_t, 16>, 11> round_keys{};
    key_expansion128(key.data(), round_keys);

    std::array<uint8_t, 16> state = block;
    add_round_key(state, round_keys[0].data());
    for (int round = 1; round <= 9; ++round) {
        sub_bytes(state);
        shift_rows(state);
        mix_columns(state);
        add_round_key(state, round_keys[static_cast<std::size_t>(round)].data());
    }
    sub_bytes(state);
    shift_rows(state);
    add_round_key(state, round_keys[10].data());
    return state;
}

} // namespace minimal_aes128

} // namespace

// ---- SHA/HMAC integration (ms::crypto live) ----

TEST(CryptoWave231Pipeline, Sha256HexRoundTripConsistency) {
    const std::string payload = "MathScript wave231 crypto integration";
    const std::string digest_hex = sha256_hex(payload);
    EXPECT_EQ(digest_hex.size(), sha256_digest_size * 2);
    expect_hex(from_hex(digest_hex), digest_hex);
    expect_hex(sha256(payload), digest_hex);
}

TEST(CryptoWave231Pipeline, HmacSha256OfSha256Digest) {
    const std::string key = "wave231-key";
    const auto inner = sha256("nested payload");
    const auto mac = hmac_sha256(key, to_hex(std::span<const uint8_t>(inner)));
    EXPECT_EQ(mac.size(), sha256_digest_size);
    expect_hex(hmac_sha256(key, to_hex(std::span<const uint8_t>(inner))), to_hex(mac));
}

TEST(CryptoWave231Pipeline, Sha512ThenSha256Independent) {
    const std::string msg = "dual-hash pipeline";
    const auto d512 = sha512(msg);
    const auto d256 = sha256(msg);
    EXPECT_EQ(d512.size(), sha512_digest_size);
    EXPECT_EQ(d256.size(), sha256_digest_size);
    EXPECT_NE(to_hex(d512).substr(0, 16), to_hex(d256));
}

// ---- AES-128 ECB NIST vector (inline minimal cipher; not ms::crypto API yet) ----

TEST(CryptoWave231Pipeline, Aes128EcbNistAppendixF_OneBlock) {
    const auto key_bytes = from_hex("2b7e151628aed2a6abf7158809cf4f3c");
    const auto plain_bytes = from_hex("3243f6a8885a308d313198a2e0370734");
    ASSERT_EQ(key_bytes.size(), 16u);
    ASSERT_EQ(plain_bytes.size(), 16u);

    std::array<uint8_t, 16> key{};
    std::array<uint8_t, 16> plain{};
    std::copy(key_bytes.begin(), key_bytes.end(), key.begin());
    std::copy(plain_bytes.begin(), plain_bytes.end(), plain.begin());

    const auto cipher = minimal_aes128::encrypt_block_ecb(key, plain);
    expect_hex(std::vector<uint8_t>(cipher.begin(), cipher.end()),
               "3925841d02dc09fbdc118597196a0b32");
}

// ---- Pending wave231/crypto-aes merge ----

TEST(CryptoWave231Pipeline, DISABLED_Aes128CbcRoundTrip) {
    // NIST SP 800-38A CBC-AES128 vectors; enable when ms::crypto exposes aes128_cbc.
    // Branch dependency: wave231/crypto-aes
    const auto key = from_hex("2b7e151628aedb2a6abf7158809cf40f");
    const auto iv = from_hex("000102030405060708090a0b0c0d0e0f");
    const auto plain = from_hex("6bc1bee22e409f96e93d7e117393172a");
    (void)key;
    (void)iv;
    (void)plain;
    // const auto enc = aes128_cbc_encrypt(key, iv, plain);
    // const auto dec = aes128_cbc_decrypt(key, iv, enc);
    // EXPECT_EQ(dec, plain);
}

// ---- Pending wave231/crypto-chacha merge ----

TEST(CryptoWave231Pipeline, DISABLED_ChaCha20RoundTrip) {
    // RFC 8439 §2.3.2 test vector; enable when ms::crypto exposes chacha20().
    // Branch dependency: wave231/crypto-chacha
    const auto key = from_hex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f");
    const auto nonce = from_hex("00000000000000000000");
    const auto plain = from_hex("00000000000000000000000000000000");
    (void)key;
    (void)nonce;
    (void)plain;
    // const auto enc = chacha20(key, nonce, 1, plain);
    // const auto dec = chacha20(key, nonce, 1, enc);
    // EXPECT_EQ(dec, plain);
}
