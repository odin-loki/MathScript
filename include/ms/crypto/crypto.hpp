// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
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
constexpr std::size_t aes_gcm_tag_size = 16;
constexpr std::size_t aes_gcm_iv_size = 12;
constexpr std::size_t chacha20_poly1305_tag_size = 16;
constexpr std::size_t chacha20_poly1305_nonce_size = 12;
constexpr std::size_t x25519_key_size = 32;
constexpr std::size_t ed25519_seed_size = 32;
constexpr std::size_t ed25519_public_key_size = 32;
constexpr std::size_t ed25519_secret_key_size = 64;
constexpr std::size_t ed25519_signature_size = 64;

using Digest256 = std::array<uint8_t, sha256_digest_size>;
using Digest512 = std::array<uint8_t, sha512_digest_size>;

std::vector<uint8_t> sha256(std::span<const uint8_t> data);
std::vector<uint8_t> sha512(std::span<const uint8_t> data);
std::vector<uint8_t> hmac_sha256(std::span<const uint8_t> key,
                                   std::span<const uint8_t> data);
std::vector<uint8_t> hmac_sha512(std::span<const uint8_t> key,
                                   std::span<const uint8_t> data);
std::vector<uint8_t> hkdf_sha256(std::span<const uint8_t> ikm,
                                 std::span<const uint8_t> salt,
                                 std::span<const uint8_t> info,
                                 std::size_t out_len);
std::vector<uint8_t> hkdf_sha512(std::span<const uint8_t> ikm,
                                 std::span<const uint8_t> salt,
                                 std::span<const uint8_t> info,
                                 std::size_t out_len);
std::vector<uint8_t> pbkdf2_hmac_sha256(std::span<const uint8_t> password,
                                        std::span<const uint8_t> salt,
                                        std::uint32_t iterations,
                                        std::size_t dklen);
std::vector<uint8_t> pbkdf2_hmac_sha512(std::span<const uint8_t> password,
                                        std::span<const uint8_t> salt,
                                        std::uint32_t iterations,
                                        std::size_t dklen);

std::vector<uint8_t> sha256(std::string_view data);
std::vector<uint8_t> sha512(std::string_view data);
std::vector<uint8_t> hmac_sha256(std::string_view key, std::string_view data);
std::vector<uint8_t> hmac_sha512(std::string_view key, std::string_view data);

std::string to_hex(std::span<const uint8_t> bytes);
std::vector<uint8_t> from_hex(std::string_view hex);
std::string sha256_hex(std::span<const uint8_t> data);
std::string sha512_hex(std::span<const uint8_t> data);
std::string hmac_sha256_hex(std::span<const uint8_t> key,
                            std::span<const uint8_t> data);
std::string hmac_sha512_hex(std::span<const uint8_t> key,
                            std::span<const uint8_t> data);

std::string sha256_hex(std::string_view data);
std::string sha512_hex(std::string_view data);
std::string hmac_sha256_hex(std::string_view key, std::string_view data);
std::string hmac_sha512_hex(std::string_view key, std::string_view data);

std::vector<uint8_t> aes128_encrypt_block(std::span<const uint8_t> key,
                                          std::span<const uint8_t> block);
std::vector<uint8_t> aes128_decrypt_block(std::span<const uint8_t> key,
                                          std::span<const uint8_t> block);
std::vector<uint8_t> aes256_encrypt_block(std::span<const uint8_t> key,
                                          std::span<const uint8_t> block);
std::vector<uint8_t> aes256_decrypt_block(std::span<const uint8_t> key,
                                          std::span<const uint8_t> block);
std::vector<uint8_t> aes128_cbc_encrypt(std::span<const uint8_t> key,
                                        std::span<const uint8_t> iv,
                                        std::span<const uint8_t> plaintext);
std::vector<uint8_t> aes128_cbc_decrypt(std::span<const uint8_t> key,
                                        std::span<const uint8_t> iv,
                                        std::span<const uint8_t> ciphertext);
std::vector<uint8_t> aes256_cbc_encrypt(std::span<const uint8_t> key,
                                        std::span<const uint8_t> iv,
                                        std::span<const uint8_t> plaintext);
std::vector<uint8_t> aes256_cbc_decrypt(std::span<const uint8_t> key,
                                        std::span<const uint8_t> iv,
                                        std::span<const uint8_t> ciphertext);

struct Aes128GcmSeal {
    std::vector<uint8_t> ciphertext;
    std::array<uint8_t, aes_gcm_tag_size> tag{};
};

Aes128GcmSeal aes128_gcm_encrypt(std::span<const uint8_t> key,
                                 std::span<const uint8_t> iv,
                                 std::span<const uint8_t> aad,
                                 std::span<const uint8_t> plaintext);
std::vector<uint8_t> aes128_gcm_decrypt(std::span<const uint8_t> key,
                                        std::span<const uint8_t> iv,
                                        std::span<const uint8_t> aad,
                                        std::span<const uint8_t> ciphertext,
                                        std::span<const uint8_t> tag);

struct Aes256GcmSeal {
    std::vector<uint8_t> ciphertext;
    std::array<uint8_t, aes_gcm_tag_size> tag{};
};

Aes256GcmSeal aes256_gcm_encrypt(std::span<const uint8_t> key,
                                 std::span<const uint8_t> iv,
                                 std::span<const uint8_t> aad,
                                 std::span<const uint8_t> plaintext);
std::vector<uint8_t> aes256_gcm_decrypt(std::span<const uint8_t> key,
                                        std::span<const uint8_t> iv,
                                        std::span<const uint8_t> aad,
                                        std::span<const uint8_t> ciphertext,
                                        std::span<const uint8_t> tag);

std::vector<uint8_t> chacha20_encrypt(const std::array<uint8_t, 32>& key,
                                      const std::array<uint8_t, 12>& nonce,
                                      std::uint32_t counter,
                                      std::span<const uint8_t> data);

struct ChaCha20Poly1305Seal {
    std::vector<uint8_t> ciphertext;
    std::array<uint8_t, chacha20_poly1305_tag_size> tag{};
};

ChaCha20Poly1305Seal chacha20_poly1305_encrypt(const std::array<uint8_t, 32>& key,
                                               const std::array<uint8_t, 12>& nonce,
                                               std::span<const uint8_t> aad,
                                               std::span<const uint8_t> plaintext);
std::vector<uint8_t> chacha20_poly1305_decrypt(const std::array<uint8_t, 32>& key,
                                               const std::array<uint8_t, 12>& nonce,
                                               std::span<const uint8_t> aad,
                                               std::span<const uint8_t> ciphertext,
                                               std::span<const uint8_t> tag);

struct X25519Keypair {
    std::array<uint8_t, x25519_key_size> private_key{};
    std::array<uint8_t, x25519_key_size> public_key{};
};

X25519Keypair x25519_keypair(std::span<const uint8_t> private_key);
std::array<uint8_t, x25519_key_size> x25519_shared_secret(std::span<const uint8_t> private_key,
                                                          std::span<const uint8_t> peer_public_key);

struct Ed25519Keypair {
    std::array<uint8_t, ed25519_secret_key_size> secret_key{};
    std::array<uint8_t, ed25519_public_key_size> public_key{};
};

Ed25519Keypair ed25519_keypair(std::span<const uint8_t> seed32);
std::array<uint8_t, ed25519_signature_size> ed25519_sign(std::span<const uint8_t> secret_or_seed,
                                                         std::span<const uint8_t> message);
bool ed25519_verify(std::span<const uint8_t> public_key, std::span<const uint8_t> message,
                    std::span<const uint8_t> signature);

bool constant_time_eq(std::span<const uint8_t> a, std::span<const uint8_t> b);

// Cryptographically secure random bytes, drawn from the operating system CSPRNG.
//
// Backend, chosen at compile time by platform macro, first available wins:
//   - Windows:      BCryptGenRandom(BCRYPT_USE_SYSTEM_PREFERRED_RNG) (bcrypt.lib)
//   - macOS/BSD:    arc4random_buf (cannot fail, so there is no error path to handle)
//   - Linux:        getrandom(2), falling back to a /dev/urandom read when the syscall
//                   is unavailable (ENOSYS on kernels < 3.17) or blocked (EPERM under
//                   seccomp)
//   - other Unix:   /dev/urandom
//   - last resort:  std::random_device, reached only when every path above failed
// Short reads and EINTR are retried; large requests are chunked so that no length
// conversion at the OS boundary can truncate. The implementation is exception-free and
// allocation-free apart from the returned vector.
//
// @param n number of bytes requested
// @return exactly n random bytes, or an EMPTY vector if the OS CSPRNG could not be
//         read — the same fail-closed convention the rest of this module uses
//         (from_hex, aes*_gcm_decrypt). A caller handling secrets must check
//         result.size() == n; a short or empty result is never padded with predictable
//         bytes. n == 0 returns an empty vector without touching the OS.
// @note Suitable for keys, nonces, IVs and salts. It is still not an HSM: the bytes come
//       from the kernel CSPRNG, not from dedicated tamper-resistant hardware, and
//       nothing here pins or zeroises the returned buffer's pages.
std::vector<uint8_t> random_bytes(std::size_t n);

// As random_bytes, but fills a caller-owned buffer and reports success explicitly
// rather than through the returned length. Same backends, same retry behaviour.
//
// @param out destination buffer; exactly out.size() bytes are written on success
// @return true on success, false if every backend failed (in which case `out` holds
//         partial or stale data and must not be used). Filling an empty span is a
//         trivial success and performs no syscall.
bool random_bytes_into(std::span<uint8_t> out);

} // namespace crypto
} // namespace ms
