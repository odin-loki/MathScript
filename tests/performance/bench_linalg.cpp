// MathScript: Linear Algebra Decomposition Benchmarks (Wave 46)
// Benchmarks for SVD, EIG, QR, LU, CHOL on various matrix sizes

#include <benchmark/benchmark.h>
#include <cmath>
#include "ms/linalg/linalg.hpp"
#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = Matrix<double>;

// Helper: create a symmetric positive definite matrix of size n
static DMatrix make_spd(size_t n) {
    DMatrix A(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        A(i, i) = static_cast<double>(n) + 2.0;  // Strong diagonal dominance
        if (i + 1 < n) {
            A(i, i + 1) = 1.0;
            A(i + 1, i) = 1.0;
        }
    }
    return A;
}

// Helper: create a random-ish square matrix of size n
static DMatrix make_general(size_t n) {
    DMatrix A(n, n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            A(i, j) = std::sin(static_cast<double>(i * n + j + 1));
    A(0, 0) += static_cast<double>(n);  // ensure non-singular
    return A;
}

// ---------------------------------------------------------------------------
// QR decomposition benchmarks
// ---------------------------------------------------------------------------

static void BM_QR_4x4(benchmark::State& state) {
    auto A = make_general(4);
    for (auto _ : state) {
        auto r = qr(A);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_QR_4x4);

static void BM_QR_8x8(benchmark::State& state) {
    auto A = make_general(8);
    for (auto _ : state) {
        auto r = qr(A);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_QR_8x8);

static void BM_QR_16x16(benchmark::State& state) {
    auto A = make_general(16);
    for (auto _ : state) {
        auto r = qr(A);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_QR_16x16);

static void BM_QR_32x32(benchmark::State& state) {
    auto A = make_general(32);
    for (auto _ : state) {
        auto r = qr(A);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_QR_32x32);

// ---------------------------------------------------------------------------
// LU decomposition benchmarks
// ---------------------------------------------------------------------------

static void BM_LU_4x4(benchmark::State& state) {
    auto A = make_general(4);
    for (auto _ : state) {
        auto r = lu(A);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_LU_4x4);

static void BM_LU_8x8(benchmark::State& state) {
    auto A = make_general(8);
    for (auto _ : state) {
        auto r = lu(A);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_LU_8x8);

static void BM_LU_16x16(benchmark::State& state) {
    auto A = make_general(16);
    for (auto _ : state) {
        auto r = lu(A);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_LU_16x16);

// ---------------------------------------------------------------------------
// Cholesky benchmarks
// ---------------------------------------------------------------------------

static void BM_Chol_4x4(benchmark::State& state) {
    auto A = make_spd(4);
    for (auto _ : state) {
        auto r = chol(A);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_Chol_4x4);

static void BM_Chol_8x8(benchmark::State& state) {
    auto A = make_spd(8);
    for (auto _ : state) {
        auto r = chol(A);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_Chol_8x8);

static void BM_Chol_16x16(benchmark::State& state) {
    auto A = make_spd(16);
    for (auto _ : state) {
        auto r = chol(A);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_Chol_16x16);

// ---------------------------------------------------------------------------
// SVD benchmarks
// ---------------------------------------------------------------------------

static void BM_SVD_4x4(benchmark::State& state) {
    auto A = make_general(4);
    for (auto _ : state) {
        auto r = svd(A);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_SVD_4x4);

static void BM_SVD_8x8(benchmark::State& state) {
    auto A = make_general(8);
    for (auto _ : state) {
        auto r = svd(A);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_SVD_8x8);

// ---------------------------------------------------------------------------
// EIG benchmarks
// ---------------------------------------------------------------------------

static void BM_Eig_4x4(benchmark::State& state) {
    auto A = make_spd(4);
    for (auto _ : state) {
        auto r = eig(A);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_Eig_4x4);

static void BM_Eig_8x8(benchmark::State& state) {
    auto A = make_spd(8);
    for (auto _ : state) {
        auto r = eig(A);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_Eig_8x8);

// ---------------------------------------------------------------------------
// Solve benchmarks (direct solve)
// ---------------------------------------------------------------------------

static void BM_Solve_4x4(benchmark::State& state) {
    auto A = make_spd(4);
    DMatrix b(4, 1, 1.0);
    for (auto _ : state) {
        auto r = solve(A, b);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_Solve_4x4);

static void BM_Solve_16x16(benchmark::State& state) {
    auto A = make_spd(16);
    DMatrix b(16, 1, 1.0);
    for (auto _ : state) {
        auto r = solve(A, b);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_Solve_16x16);

BENCHMARK_MAIN();
