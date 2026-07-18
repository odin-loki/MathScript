#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ms {
namespace crypto {

constexpr std::size_t sha256_digest_size = 32;
constexpr std::size_t sha512_digest_size = 64;
constexpr std::size_t aes_block_size = 16;
constexpr std::size_t aes128_key_size = 16;
constexpr std::size_t aes256_key_size = 32;

using Digest256 = std::array<uint8_t, sha256_digest_size>;
using Digest512 = std::array<uint8_t, sha512_digest_size>;

std::vector<uint8_t> sha256(std::span<const uint8_t> data);
std::vector<uint8_t> sha512(std::span<const uint8_t> data);
std::vector<uint8_t> hmac_sha256(std::span<const uint8_t> key,
                                   std::span<const uint8_t> data);

std::vector<uint8_t> sha256(std::string_view data);
std::vector<uint8_t> sha512(std::string_view data);
std::vector<uint8_t> hmac_sha256(std::string_view key, std::string_view data);

std::string to_hex(std::span<const uint8_t> bytes);
std::string sha256_hex(std::span<const uint8_t> data);
std::string sha512_hex(std::span<const uint8_t> data);
std::string hmac_sha256_hex(std::span<const uint8_t> key,
                            std::span<const uint8_t> data);

std::string sha256_hex(std::string_view data);
std::string sha512_hex(std::string_view data);
std::string hmac_sha256_hex(std::string_view key, std::string_view data);

std::vector<uint8_t> aes128_encrypt_block(std::span<const uint8_t> key,
                                          std::span<const uint8_t> block);
std::vector<uint8_t> aes256_encrypt_block(std::span<const uint8_t> key,
                                          std::span<const uint8_t> block);
std::vector<uint8_t> aes128_cbc_encrypt(std::span<const uint8_t> key,
                                        std::span<const uint8_t> iv,
                                        std::span<const uint8_t> plaintext);
std::vector<uint8_t> aes128_cbc_decrypt(std::span<const uint8_t> key,
                                        std::span<const uint8_t> iv,
                                        std::span<const uint8_t> ciphertext);

std::vector<uint8_t> chacha20_encrypt(const std::array<uint8_t, 32>& key,
                                      const std::array<uint8_t, 12>& nonce,
                                      std::uint32_t counter,
                                      std::span<const uint8_t> data);

} // namespace crypto
} // namespace ms
