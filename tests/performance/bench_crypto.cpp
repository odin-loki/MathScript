// MathScript crypto benchmark suite

#include <benchmark/benchmark.h>
#include <cstdint>
#include <vector>

#include "ms/crypto/crypto.hpp"

static void BM_Sha256_1MB(benchmark::State& state) {
    constexpr std::size_t kSize = 1024 * 1024;
    const std::vector<std::uint8_t> data(kSize, 0x42);
    for (auto _ : state) {
        const auto digest = ms::crypto::sha256(data);
        benchmark::DoNotOptimize(digest.data());
    }
    state.SetBytesProcessed(state.iterations() * kSize);
}

BENCHMARK(BM_Sha256_1MB);
