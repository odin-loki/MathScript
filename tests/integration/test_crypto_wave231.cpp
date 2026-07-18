// Wave 231 — ms::crypto integration tests (SHA/HMAC + AES/ChaCha round-trip pipelines).

#include "ms/crypto/crypto.hpp"

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

std::array<uint8_t, 32> key32_from_hex(std::string_view hex) {
    const auto bytes = from_hex(hex);
    std::array<uint8_t, 32> key{};
    std::copy(bytes.begin(), bytes.end(), key.begin());
    return key;
}

std::array<uint8_t, 12> nonce12_from_hex(std::string_view hex) {
    const auto bytes = from_hex(hex);
    std::array<uint8_t, 12> nonce{};
    std::copy(bytes.begin(), bytes.end(), nonce.begin());
    return nonce;
}

} // namespace

TEST(CryptoWave231Pipeline, Sha256AndSha512DifferentDigests) {
    const std::string msg = "MathScript Wave 231 integration";
    const auto d512 = sha512(msg);
    const auto d256 = sha256(msg);
    EXPECT_EQ(d512.size(), sha512_digest_size);
    EXPECT_EQ(d256.size(), sha256_digest_size);
    EXPECT_NE(to_hex(d512).substr(0, 16), to_hex(d256));
}

TEST(CryptoWave231Pipeline, Aes128EcbNistAppendixF_OneBlock) {
    const auto key = from_hex("2b7e151628aed2a6abf7158809cf4f3c");
    const auto plain = from_hex("3243f6a8885a308d313198a2e0370734");
    expect_hex(aes128_encrypt_block(key, plain), "3925841d02dc09fbdc118597196a0b32");
}

TEST(CryptoWave231Pipeline, Aes128CbcRoundTrip) {
    const auto key = from_hex("2b7e151628aedb2a6abf7158809cf40f");
    const auto iv = from_hex("000102030405060708090a0b0c0d0e0f");
    const auto plain = from_hex("6bc1bee22e409f96e93d7e117393172a");
    const auto enc = aes128_cbc_encrypt(key, iv, plain);
    const auto dec = aes128_cbc_decrypt(key, iv, enc);
    EXPECT_EQ(dec, plain);
}

TEST(CryptoWave231Pipeline, ChaCha20RoundTrip) {
    const auto key = key32_from_hex(
        "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f");
    const auto nonce = nonce12_from_hex("00000000000000000000");
    const auto plain = from_hex("00000000000000000000000000000000");
    const auto enc = chacha20_encrypt(key, nonce, 1, plain);
    const auto dec = chacha20_encrypt(key, nonce, 1, enc);
    EXPECT_EQ(dec, plain);
}
