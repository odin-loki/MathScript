// MathScript Benchmark: compression hot paths (RLE, Huffman, LZ77)

#include <benchmark/benchmark.h>
#include <cstdint>
#include <string>
#include <vector>

#include "ms/compress/compress.hpp"

using ms::compress::Bytes;
using namespace ms::compress;

namespace {

Bytes make_rle_friendly_buffer(std::size_t n) {
    Bytes data;
    data.reserve(n);
    uint8_t v = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if ((i & 3u) == 0) ++v;
        data.push_back(v);
    }
    return data;
}

Bytes make_huffman_skewed_buffer(std::size_t n) {
    static const std::string kChunk =
        "the quick brown fox jumps over the lazy dog. "
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    Bytes data;
    data.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        data.push_back(static_cast<uint8_t>(kChunk[i % kChunk.size()]));
    return data;
}

Bytes make_lz77_repetitive_buffer(std::size_t n) {
    static const std::string kPattern = "abracadabraXY";
    Bytes data;
    data.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        data.push_back(static_cast<uint8_t>(kPattern[i % kPattern.size()]));
    return data;
}

} // namespace

static void BM_RLE(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto data = make_rle_friendly_buffer(n);
    for (auto _ : state) {
        benchmark::DoNotOptimize(rle_encode(data));
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}
BENCHMARK(BM_RLE)->Arg(65536)->Arg(262144)->Arg(1048576);

static void BM_Huffman(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto data = make_huffman_skewed_buffer(n);
    for (auto _ : state) {
        benchmark::DoNotOptimize(huffman_encode(data));
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}
BENCHMARK(BM_Huffman)->Arg(65536)->Arg(262144)->Arg(1048576);

static void BM_LZ77(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const auto data = make_lz77_repetitive_buffer(n);
    for (auto _ : state) {
        benchmark::DoNotOptimize(lz77_encode(data));
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n));
}
BENCHMARK(BM_LZ77)->Arg(65536)->Arg(262144)->Arg(1048576);

BENCHMARK_MAIN();
