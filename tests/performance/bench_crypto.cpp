// MathScript crypto benchmark suite

#include <benchmark/benchmark.h>
#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "ms/crypto/crypto.hpp"

using namespace ms::crypto;

namespace {

constexpr std::size_t kSha256InputBytes = 1024 * 1024;
constexpr std::size_t kPayloadBytes = 64 * 1024;
constexpr std::size_t kAesBlockBytes = aes_block_size;

std::vector<uint8_t> make_byte_buffer(std::size_t n, uint8_t seed = 0) {
    std::vector<uint8_t> data(n);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = static_cast<uint8_t>((seed + i * 131) & 0xFF);
    }
    return data;
}

} // namespace

static void BM_Sha256_1MB(benchmark::State& state) {
    const auto data = make_byte_buffer(kSha256InputBytes);
    for (auto _ : state) {
        auto digest = sha256(std::span<const uint8_t>(data));
        benchmark::DoNotOptimize(digest.data());
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(kSha256InputBytes));
}
BENCHMARK(BM_Sha256_1MB);

static void BM_HmacSha256(benchmark::State& state) {
    const auto key = make_byte_buffer(32, 7);
    const auto data = make_byte_buffer(4096, 19);
    for (auto _ : state) {
        auto digest = hmac_sha256(std::span<const uint8_t>(key),
                                  std::span<const uint8_t>(data));
        benchmark::DoNotOptimize(digest.data());
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(data.size()));
}
BENCHMARK(BM_HmacSha256);

static void BM_HmacSha512(benchmark::State& state) {
    const auto key = make_byte_buffer(32, 7);
    const auto data = make_byte_buffer(4096, 19);
    for (auto _ : state) {
        auto digest = hmac_sha512(std::span<const uint8_t>(key),
                                  std::span<const uint8_t>(data));
        benchmark::DoNotOptimize(digest.data());
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(data.size()));
}
BENCHMARK(BM_HmacSha512);

static void BM_HkdfSha256(benchmark::State& state) {
    const auto ikm = make_byte_buffer(22, 11);
    const auto salt = make_byte_buffer(13, 17);
    const auto info = make_byte_buffer(10, 23);
    constexpr std::size_t kOutLen = 32;
    for (auto _ : state) {
        auto okm = hkdf_sha256(std::span<const uint8_t>(ikm),
                               std::span<const uint8_t>(salt),
                               std::span<const uint8_t>(info),
                               kOutLen);
        benchmark::DoNotOptimize(okm.data());
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(kOutLen));
}
BENCHMARK(BM_HkdfSha256);

static void BM_Pbkdf2HmacSha256(benchmark::State& state) {
    const auto password = make_byte_buffer(24, 29);
    const auto salt = make_byte_buffer(16, 31);
    constexpr std::uint32_t kIterations = 1000;
    constexpr std::size_t kDkLen = 32;
    for (auto _ : state) {
        auto dk = pbkdf2_hmac_sha256(std::span<const uint8_t>(password),
                                     std::span<const uint8_t>(salt),
                                     kIterations,
                                     kDkLen);
        benchmark::DoNotOptimize(dk.data());
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(kDkLen));
}
BENCHMARK(BM_Pbkdf2HmacSha256);

static void BM_X25519SharedSecret(benchmark::State& state) {
    const auto private_key = make_byte_buffer(x25519_key_size, 37);
    const auto peer_public_key = make_byte_buffer(x25519_key_size, 41);
    for (auto _ : state) {
        auto shared = x25519_shared_secret(std::span<const uint8_t>(private_key),
                                           std::span<const uint8_t>(peer_public_key));
        benchmark::DoNotOptimize(shared.data());
    }
}
BENCHMARK(BM_X25519SharedSecret);

static void BM_Aes128EncryptBlock(benchmark::State& state) {
    const auto key = make_byte_buffer(aes128_key_size, 31);
    const auto block = make_byte_buffer(kAesBlockBytes, 43);
    for (auto _ : state) {
        auto ciphertext = aes128_encrypt_block(std::span<const uint8_t>(key),
                                               std::span<const uint8_t>(block));
        benchmark::DoNotOptimize(ciphertext.data());
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(kAesBlockBytes));
}
BENCHMARK(BM_Aes128EncryptBlock)->MinTime(0.001);

static void BM_Aes128CbcEncrypt_64KB(benchmark::State& state) {
    const auto key = make_byte_buffer(aes128_key_size, 53);
    const auto iv = make_byte_buffer(aes_block_size, 61);
    const auto plaintext = make_byte_buffer(kPayloadBytes, 67);
    for (auto _ : state) {
        auto ciphertext = aes128_cbc_encrypt(std::span<const uint8_t>(key),
                                             std::span<const uint8_t>(iv),
                                             std::span<const uint8_t>(plaintext));
        benchmark::DoNotOptimize(ciphertext.data());
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(kPayloadBytes));
}
BENCHMARK(BM_Aes128CbcEncrypt_64KB)->MinTime(0.001);

static void BM_ChaCha20Encrypt_64KB(benchmark::State& state) {
    const auto key_bytes = make_byte_buffer(32, 71);
    const auto nonce_bytes = make_byte_buffer(12, 79);
    std::array<uint8_t, 32> key{};
    std::array<uint8_t, 12> nonce{};
    for (std::size_t i = 0; i < key.size(); ++i) {
        key[i] = key_bytes[i];
    }
    for (std::size_t i = 0; i < nonce.size(); ++i) {
        nonce[i] = nonce_bytes[i];
    }
    const auto plaintext = make_byte_buffer(kPayloadBytes, 83);
    for (auto _ : state) {
        auto ciphertext =
            chacha20_encrypt(key, nonce, 1, std::span<const uint8_t>(plaintext));
        benchmark::DoNotOptimize(ciphertext.data());
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(kPayloadBytes));
}
BENCHMARK(BM_ChaCha20Encrypt_64KB)->MinTime(0.001);

static void BM_Ed25519Sign(benchmark::State& state) {
    const auto seed = make_byte_buffer(32, 97);
    const auto kp = ed25519_keypair(std::span<const uint8_t>(seed));
    const auto msg = make_byte_buffer(32, 101);
    for (auto _ : state) {
        auto sig = ed25519_sign(kp.secret_key, std::span<const uint8_t>(msg));
        benchmark::DoNotOptimize(sig.data());
    }
}
BENCHMARK(BM_Ed25519Sign)->MinTime(0.001);

static void BM_Ed25519Verify(benchmark::State& state) {
    const auto seed = make_byte_buffer(32, 97);
    const auto kp = ed25519_keypair(std::span<const uint8_t>(seed));
    const auto msg = make_byte_buffer(32, 101);
    const auto sig = ed25519_sign(kp.secret_key, std::span<const uint8_t>(msg));
    for (auto _ : state) {
        const bool ok =
            ed25519_verify(kp.public_key, std::span<const uint8_t>(msg), sig);
        benchmark::DoNotOptimize(ok);
    }
}
BENCHMARK(BM_Ed25519Verify)->MinTime(0.001);

BENCHMARK_MAIN();
