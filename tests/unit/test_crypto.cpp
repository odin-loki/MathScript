#include "ms/crypto/crypto.hpp"

#include <gtest/gtest.h>

#include <array>
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

void expect_hex(const std::vector<uint8_t>& digest, std::string_view expected) {
    EXPECT_EQ(to_hex(std::span<const uint8_t>(digest)), expected);
}

} // namespace

// ---- SHA-256 (FIPS 180-4 / NIST example values) ----

TEST(CryptoSha256, EmptyString) {
    expect_hex(sha256(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST(CryptoSha256, Abc) {
    expect_hex(sha256("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(CryptoSha256, LongMessage448Bits) {
    const std::string msg =
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    expect_hex(sha256(msg),
               "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST(CryptoSha256, MillionA) {
    const std::string msg(1'000'000, 'a');
    expect_hex(sha256(msg),
               "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(CryptoSha256, DigestSize) {
    EXPECT_EQ(sha256("test").size(), sha256_digest_size);
}

TEST(CryptoSha256, HexHelperMatchesRaw) {
    const std::string msg = "MathScript crypto wave 220";
    EXPECT_EQ(sha256_hex(msg), to_hex(sha256(msg)));
}

TEST(CryptoSha256, SpanInput) {
    const std::array<uint8_t, 3> bytes = {0x61, 0x62, 0x63};
    expect_hex(sha256(bytes), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

// ---- SHA-512 (FIPS 180-4 / NIST example values) ----

TEST(CryptoSha512, EmptyString) {
    expect_hex(sha512(""),
               "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce"
               "47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
}

TEST(CryptoSha512, Abc) {
    expect_hex(sha512("abc"),
               "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
               "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f");
}

TEST(CryptoSha512, LongMessage896Bits) {
    const std::string msg =
        "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"
        "ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
    expect_hex(sha512(msg),
               "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb688901"
               "8501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909");
}

TEST(CryptoSha512, DigestSize) {
    EXPECT_EQ(sha512("wave220").size(), sha512_digest_size);
}

TEST(CryptoSha512, HexHelperMatchesRaw) {
    const std::string msg = "sha512 hex roundtrip";
    EXPECT_EQ(sha512_hex(msg), to_hex(sha512(msg)));
}

// ---- HMAC-SHA256 (RFC 4231 test vectors) ----

TEST(CryptoHmacSha256, TestCase1) {
    const std::vector<uint8_t> key(20, 0x0b);
    const std::string data = "Hi There";
    expect_hex(hmac_sha256(std::span<const uint8_t>(key),
                           std::span<const uint8_t>(
                               reinterpret_cast<const uint8_t*>(data.data()),
                               data.size())),
               "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
}

TEST(CryptoHmacSha256, TestCase2) {
    const std::string key = "Jefe";
    const std::string data = "what do ya want for nothing?";
    expect_hex(hmac_sha256(key, data),
               "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
}

TEST(CryptoHmacSha256, TestCase3) {
    const auto key = from_hex(std::string(40, 'a'));
    const auto data = from_hex(std::string(100, 'd'));
    expect_hex(hmac_sha256(std::span<const uint8_t>(key), std::span<const uint8_t>(data)),
               "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe");
}

TEST(CryptoHmacSha256, TestCase4_TruncatedKey) {
    const auto key = from_hex(std::string(50, 'a'));
    const auto data = from_hex(std::string(100, 'd'));
    expect_hex(hmac_sha256(std::span<const uint8_t>(key), std::span<const uint8_t>(data)),
               "6659dd1f80346d72aeca625366776e432c993eea9e1c4e979b508821b75ab528");
}

TEST(CryptoHmacSha256, TestCase5_LargerKey) {
    const auto key = from_hex(std::string(262, 'a'));
    const auto data = from_hex(std::string(100, 'd'));
    expect_hex(hmac_sha256(std::span<const uint8_t>(key), std::span<const uint8_t>(data)),
               "124c7d2385aa1743aaad12204e3464f06305fd1a6d291250fa564dceffab0c8a");
}

TEST(CryptoHmacSha256, HexHelperMatchesRaw) {
    const std::string key = "secret";
    const std::string data = "payload";
    EXPECT_EQ(hmac_sha256_hex(key, data), to_hex(hmac_sha256(key, data)));
}

TEST(CryptoToHex, Empty) {
    EXPECT_EQ(to_hex(std::vector<uint8_t>{}), "");
}

TEST(CryptoToHex, LeadingZeroNibble) {
    EXPECT_EQ(to_hex(std::vector<uint8_t>{0x0a, 0x0b}), "0a0b");
}
