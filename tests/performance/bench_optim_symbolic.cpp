// MathScript: Benchmarks for Optimization and Symbolic modules

#include <benchmark/benchmark.h>
#include <cmath>
#include <vector>

#include "ms/optim/optim.hpp"
#include "ms/symbolic/symbolic.hpp"
#include "ms/poly/poly.hpp"
#include "ms/domain/domain.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// Optimization benchmarks
// ---------------------------------------------------------------------------

static void BM_GoldenSection(benchmark::State& state) {
    auto f = [](double x) { return (x - 2.5) * (x - 2.5) + 1.0; };
    for (auto _ : state) {
        double r = golden_section(f, 0.0, 5.0, 1e-8);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_GoldenSection);

static void BM_Newton1D(benchmark::State& state) {
    auto f = [](double x) { return (x - 3.0) * (x - 3.0); };
    for (auto _ : state) {
        double r = newton_1d(f, 4.0, 1e-10, 100);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_Newton1D);

static void BM_GradientDescent(benchmark::State& state) {
    auto f = [](double x, double y) { return x * x + y * y; };
    for (auto _ : state) {
        auto r = gradient_descent(f, 1.0, 1.0, 0.1, 500);
        benchmark::DoNotOptimize(r.f_val);
    }
}
BENCHMARK(BM_GradientDescent);

static void BM_NewtonRaphson(benchmark::State& state) {
    auto f = [](double x, double y) { return x * x + y * y - 4.0; };
    for (auto _ : state) {
        auto [x, y] = newton_raphson(f, 1.0, 1.0, 1e-10, 50);
        benchmark::DoNotOptimize(x);
    }
}
BENCHMARK(BM_NewtonRaphson);

static void BM_Broyden(benchmark::State& state) {
    auto f = [](double x, double /*y*/) { return x * x - 4.0; };
    for (auto _ : state) {
        auto [x, y] = broyden(f, 1.5, 0.0);
        benchmark::DoNotOptimize(x);
    }
}
BENCHMARK(BM_Broyden);

static void BM_SimplexSolver(benchmark::State& state) {
    std::vector<double> c = {1.0, 1.0};
    std::vector<std::vector<double>> A = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    std::vector<double> b = {3.0, 3.0, 5.0};
    for (auto _ : state) {
        auto [x, obj] = simplex_solver(c, A, b);
        benchmark::DoNotOptimize(obj);
    }
}
BENCHMARK(BM_SimplexSolver);

// ---------------------------------------------------------------------------
// Symbolic math benchmarks
// ---------------------------------------------------------------------------

static void BM_SymEval_Simple(benchmark::State& state) {
    for (auto _ : state) {
        auto expr = sym_add(sym_pow(sym_var("x"), sym_const(2.0)), sym_const(-4.0));
        double v = sym_eval(expr, {{"x", 3.0}});
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_SymEval_Simple);

static void BM_SymDiff(benchmark::State& state) {
    for (auto _ : state) {
        auto expr = sym_add(sym_pow(sym_var("x"), sym_const(3.0)), sym_mul(sym_const(2.0), sym_var("x")));
        auto deriv = sym_diff(std::move(expr), "x");
        benchmark::DoNotOptimize(&deriv);
    }
}
BENCHMARK(BM_SymDiff);

static void BM_SymSimplify(benchmark::State& state) {
    for (auto _ : state) {
        auto expr = sym_add(sym_mul(sym_const(0.0), sym_var("x")), sym_const(5.0));
        auto simplified = sym_simplify(std::move(expr));
        benchmark::DoNotOptimize(&simplified);
    }
}
BENCHMARK(BM_SymSimplify);

static void BM_SymParse(benchmark::State& state) {
    const std::string text = "x^2 + sin(y) - 3*x + 2.5e-3";
    for (auto _ : state) {
        auto parsed = sym_parse(text);
        benchmark::DoNotOptimize(parsed);
    }
}
BENCHMARK(BM_SymParse);

// ---------------------------------------------------------------------------
// Polynomial benchmarks
// ---------------------------------------------------------------------------

static void BM_PolyEval_Degree10(benchmark::State& state) {
    std::vector<double> coeffs = {1.0, -2.0, 3.0, -1.0, 0.5, 2.0, -1.5, 0.3, 1.1, -0.7, 0.2};
    for (auto _ : state) {
        auto r = poly_eval(coeffs, 2.5);
        benchmark::DoNotOptimize(r.data());
    }
}
BENCHMARK(BM_PolyEval_Degree10);

static void BM_PolyMul(benchmark::State& state) {
    std::vector<double> a = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> b = {1.0, -1.0, 2.0, -2.0, 3.0};
    for (auto _ : state) {
        auto r = poly_mul(a, b);
        benchmark::DoNotOptimize(r.data());
    }
}
BENCHMARK(BM_PolyMul);

static void BM_PolyDeriv(benchmark::State& state) {
    std::vector<double> coeffs(20);
    for (size_t i = 0; i < 20; ++i) coeffs[i] = static_cast<double>(i + 1);
    for (auto _ : state) {
        auto r = poly_deriv(coeffs);
        benchmark::DoNotOptimize(r.data());
    }
}
BENCHMARK(BM_PolyDeriv);

// ---------------------------------------------------------------------------
// Domain benchmarks
// ---------------------------------------------------------------------------

static void BM_Factorial(benchmark::State& state) {
    const size_t n = static_cast<size_t>(state.range(0));
    for (auto _ : state) {
        auto r = factorial(n);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_Factorial)->Arg(10)->Arg(15)->Arg(18);

static void BM_NChooseK(benchmark::State& state) {
    const size_t n = static_cast<size_t>(state.range(0));
    for (auto _ : state) {
        auto r = nchoosek(n, n / 2);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_NChooseK)->Arg(10)->Arg(20)->Arg(30);

static void BM_GCD(benchmark::State& state) {
    const size_t a = 123456789ULL;
    const size_t b = 987654321ULL;
    for (auto _ : state) {
        auto r = gcd(a, b);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_GCD);

BENCHMARK_MAIN();
