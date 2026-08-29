// MathScript Benchmark: quantum density-matrix ops + bignum hot paths

#include <benchmark/benchmark.h>
#include <cmath>
#include <string>

#include "ms/bignum/bignum.hpp"
#include "ms/quantum/quantum.hpp"

using ms::bignum::BigInt;
using ms::bignum::bigint_pow_mod;
using ms::quantum::C;
using ms::quantum::DensityMatrix;
using ms::quantum::commutator;
using ms::quantum::matmul_dm;
using ms::quantum::qft_gate;

namespace {

DensityMatrix make_random_dm(int n, int seed) {
    DensityMatrix A(n, std::vector<C>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const double re = std::sin(static_cast<double>(seed + i * n + j + 1));
            const double im = std::cos(static_cast<double>(seed + i * n + j + 2));
            A[i][j] = C(re, im);
        }
    return A;
}

std::string make_big_decimal_string(int digits) {
    std::string s;
    s.reserve(static_cast<size_t>(digits));
    s += '9';
    for (int i = 1; i < digits; ++i)
        s += static_cast<char>('0' + (i * 7 + 3) % 10);
    return s;
}

} // namespace

static void BM_MatmulDM(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const auto A = make_random_dm(n, 11);
    const auto B = make_random_dm(n, 23);
    for (auto _ : state) {
        auto Cm = matmul_dm(A, B);
        benchmark::DoNotOptimize(Cm);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) * n * n);
}
BENCHMARK(BM_MatmulDM)->Arg(8)->Arg(16)->Arg(32);

static void BM_Commutator(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const auto X = make_random_dm(n, 5);
    const auto Z = make_random_dm(n, 7);
    for (auto _ : state) {
        auto Cm = commutator(X, Z);
        benchmark::DoNotOptimize(Cm);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n) * n * n);
}
BENCHMARK(BM_Commutator)->Arg(8)->Arg(16);

static void BM_QFTGate(benchmark::State& state) {
    const int qubits = static_cast<int>(state.range(0));
    for (auto _ : state) {
        auto Q = qft_gate(qubits);
        benchmark::DoNotOptimize(Q);
    }
    const int N = 1 << qubits;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(N) * N);
}
BENCHMARK(BM_QFTGate)->Arg(4)->Arg(5)->Arg(6);

static void BM_BigIntMultiply(benchmark::State& state) {
    const int digits = static_cast<int>(state.range(0));
    const BigInt a(make_big_decimal_string(digits));
    const BigInt b(make_big_decimal_string(digits + 1));
    for (auto _ : state) {
        auto p = a * b;
        benchmark::DoNotOptimize(p);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(digits));
}
BENCHMARK(BM_BigIntMultiply)->Arg(100)->Arg(500);

static void BM_BigIntPowMod(benchmark::State& state) {
    const BigInt base("123456789");
    const BigInt exp("98765432109876543210");
    const BigInt mod("999999937");
    for (auto _ : state) {
        auto r = bigint_pow_mod(base, exp, mod);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_BigIntPowMod);

BENCHMARK_MAIN();
