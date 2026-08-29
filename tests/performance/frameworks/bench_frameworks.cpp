#include <benchmark/benchmark.h>
#include <cstdint>
#include <array>
#include <vector>
#include <span>
#include "ms/frameworks/gria/gria.hpp"
#include "ms/frameworks/izaac/izaac.hpp"

// ---------------------------------------------------------------------------
// Gria – entropy
// ---------------------------------------------------------------------------

static void BM_GriaEntropy(benchmark::State& state) {
    std::vector<double> data(100);
    for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<double>(i % 16);

    for (auto _ : state)
        benchmark::DoNotOptimize(ms::gria::entropy(data));
}
BENCHMARK(BM_GriaEntropy);

static void BM_GriaEntropy_Large(benchmark::State& state) {
    std::vector<double> data(10000);
    for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<double>(i % 64);

    for (auto _ : state)
        benchmark::DoNotOptimize(ms::gria::entropy(data));
}
BENCHMARK(BM_GriaEntropy_Large);

// ---------------------------------------------------------------------------
// Gria – matrix_alpha
// ---------------------------------------------------------------------------

static void BM_GriaMatrixAlpha_4x4(benchmark::State& state) {
    ms::Matrix<double> x(4, 4);
    ms::Matrix<double> fx(4, 4);
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            x(r, c)  = static_cast<double>(r * 4 + c + 1);
            fx(r, c) = static_cast<double>((r * 4 + c + 1) * 2);
        }
    }

    for (auto _ : state)
        benchmark::DoNotOptimize(ms::gria::matrix_alpha(x, fx));
}
BENCHMARK(BM_GriaMatrixAlpha_4x4);

// ---------------------------------------------------------------------------
// Gria – classify
// ---------------------------------------------------------------------------

static void BM_GriaClassify(benchmark::State& state) {
    double alpha = 0.03;
    for (auto _ : state)
        benchmark::DoNotOptimize(ms::gria::classify(alpha));
}
BENCHMARK(BM_GriaClassify);

// ---------------------------------------------------------------------------
// GF(2^n) arithmetic
// ---------------------------------------------------------------------------

static void BM_GF2N_Mul(benchmark::State& state) {
    constexpr uint64_t poly = 0x11B; // AES irreducible polynomial for GF(2^8)
    uint64_t a = 0x53;
    uint64_t b = 0xCA;
    for (auto _ : state) {
        for (int i = 0; i < 1000; ++i) {
            benchmark::DoNotOptimize(ms::gria::gf2n::mul(a, b, poly));
            a = (a + 1) & 0xFF;
        }
    }
}
BENCHMARK(BM_GF2N_Mul);

static void BM_GF2N_Pow(benchmark::State& state) {
    constexpr uint64_t poly = 0x11B;
    for (auto _ : state)
        benchmark::DoNotOptimize(ms::gria::gf2n::pow(0x02, 255, poly));
}
BENCHMARK(BM_GF2N_Pow);

static void BM_GF2N_GenerateField(benchmark::State& state) {
    for (auto _ : state)
        benchmark::DoNotOptimize(ms::gria::gf2n::generate_field(8));
}
BENCHMARK(BM_GF2N_GenerateField);

// ---------------------------------------------------------------------------
// Cellular automata
// ---------------------------------------------------------------------------

static void BM_GriaCAStep(benchmark::State& state) {
    std::vector<uint8_t> ca_state(64, 0);
    ca_state[32] = 1;
    constexpr uint8_t rule = 30;

    for (auto _ : state)
        benchmark::DoNotOptimize(ms::gria::ca::step(ca_state, rule));
}
BENCHMARK(BM_GriaCAStep);

static void BM_GriaAlphaCA(benchmark::State& state) {
    for (auto _ : state)
        benchmark::DoNotOptimize(ms::gria::ca::alpha_ca(30, 100, 32));
}
BENCHMARK(BM_GriaAlphaCA);

// ---------------------------------------------------------------------------
// LFSR
// ---------------------------------------------------------------------------

static void BM_GriaLFSRStep(benchmark::State& state) {
    constexpr uint64_t poly = 0xB8;
    uint64_t lfsr_state = 0x01;
    for (auto _ : state) {
        for (int i = 0; i < 1000; ++i) {
            lfsr_state = ms::gria::lfsr::step(lfsr_state, poly);
            benchmark::DoNotOptimize(lfsr_state);
        }
    }
}
BENCHMARK(BM_GriaLFSRStep);

static void BM_GriaAlphaLFSR(benchmark::State& state) {
    for (auto _ : state)
        benchmark::DoNotOptimize(ms::gria::lfsr::alpha_lfsr(0xB8, 100));
}
BENCHMARK(BM_GriaAlphaLFSR);

// ---------------------------------------------------------------------------
// Izaac – CSPRNG
// ---------------------------------------------------------------------------

static void BM_IzaacCSPRNG_NextU64(benchmark::State& state) {
    std::array<uint8_t, 32> seed{};
    for (std::size_t i = 0; i < seed.size(); ++i)
        seed[i] = static_cast<uint8_t>(i);
    ms::izaac::CSPRNG rng(seed);

    for (auto _ : state) {
        for (int i = 0; i < 1000; ++i)
            benchmark::DoNotOptimize(rng.next_u64());
    }
}
BENCHMARK(BM_IzaacCSPRNG_NextU64);

static void BM_IzaacCSPRNG_Fill_64(benchmark::State& state) {
    std::array<uint8_t, 32> seed{};
    ms::izaac::CSPRNG rng(seed);
    std::vector<uint8_t> buf(64);

    for (auto _ : state) {
        rng.fill(buf);
        benchmark::DoNotOptimize(buf.data());
    }
}
BENCHMARK(BM_IzaacCSPRNG_Fill_64);

static void BM_IzaacCSPRNG_Fill_4096(benchmark::State& state) {
    std::array<uint8_t, 32> seed{};
    ms::izaac::CSPRNG rng(seed);
    std::vector<uint8_t> buf(4096);

    for (auto _ : state) {
        rng.fill(buf);
        benchmark::DoNotOptimize(buf.data());
    }
}
BENCHMARK(BM_IzaacCSPRNG_Fill_4096);

// ---------------------------------------------------------------------------
// Izaac – random matrices
// ---------------------------------------------------------------------------

static void BM_IzaacRandMatrix_4x4(benchmark::State& state) {
    for (auto _ : state)
        benchmark::DoNotOptimize(ms::izaac::rand_matrix(4, 4));
}
BENCHMARK(BM_IzaacRandMatrix_4x4);

static void BM_IzaacRandMatrix_64x64(benchmark::State& state) {
    for (auto _ : state)
        benchmark::DoNotOptimize(ms::izaac::rand_matrix(64, 64));
}
BENCHMARK(BM_IzaacRandMatrix_64x64);

// ---------------------------------------------------------------------------
// Izaac – Monte Carlo pi estimation
// ---------------------------------------------------------------------------

static void BM_IzaacEstimatePi(benchmark::State& state) {
    std::array<uint8_t, 32> seed{};
    ms::izaac::CSPRNG rng(seed);

    for (auto _ : state)
        benchmark::DoNotOptimize(ms::izaac::mc::estimate_pi(1000, rng));
}
BENCHMARK(BM_IzaacEstimatePi);

// ---------------------------------------------------------------------------
// Izaac – VRF keygen
// ---------------------------------------------------------------------------

static void BM_IzaacKeygen(benchmark::State& state) {
    for (auto _ : state)
        benchmark::DoNotOptimize(ms::izaac::keygen());
}
BENCHMARK(BM_IzaacKeygen);

// ---------------------------------------------------------------------------
// Izaac – consensus election simulation
// ---------------------------------------------------------------------------

static void BM_IzaacConsensus_Election(benchmark::State& state) {
    ms::izaac::consensus::Cluster cluster(7, 42);
    for (auto _ : state)
        benchmark::DoNotOptimize(cluster.run_election());
}
BENCHMARK(BM_IzaacConsensus_Election);

static void BM_IzaacConsensus_Replicate(benchmark::State& state) {
    ms::izaac::consensus::Cluster cluster(7, 42);
    const int leader = cluster.run_election();
    for (auto _ : state)
        benchmark::DoNotOptimize(cluster.replicate(leader, "cmd"));
}
BENCHMARK(BM_IzaacConsensus_Replicate);

// ---------------------------------------------------------------------------
// Gria – CA divergence trajectory
// ---------------------------------------------------------------------------

static void BM_GriaCADivergence(benchmark::State& state) {
    std::vector<uint8_t> a(64, 0);
    std::vector<uint8_t> b(64, 0);
    a[32] = 1;
    b[33] = 1;
    for (auto _ : state)
        benchmark::DoNotOptimize(ms::gria::ca::divergence_trajectory(a, b, 30, 50));
}
BENCHMARK(BM_GriaCADivergence);

// ---------------------------------------------------------------------------

BENCHMARK_MAIN();
