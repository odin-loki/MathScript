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

BENCHMARK_MAIN();
