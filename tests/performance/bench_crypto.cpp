// MathScript crypto benchmark suite

#include <benchmark/benchmark.h>
#include <cstdint>
#include <span>
#include <vector>

#include "ms/crypto/crypto.hpp"

using namespace ms::crypto;

namespace {

constexpr std::size_t kSha256InputBytes = 1024 * 1024;

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

BENCHMARK_MAIN();
