// MathScript Benchmark: Probability Distribution Functions
// Benchmarks for norm_pdf/cdf/ppf, t_pdf/cdf, gamma_pdf, binom_pdf/cdf, pois_pdf/cdf, chi2, uniform

#include <benchmark/benchmark.h>
#include <cmath>
#include "ms/prob/prob.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// Normal distribution
// ---------------------------------------------------------------------------

static void BM_NormPDF(benchmark::State& state) {
    for (auto _ : state) {
        for (double x = -3.0; x <= 3.0; x += 0.1) {
            benchmark::DoNotOptimize(norm_pdf(x, 0.0, 1.0));
        }
    }
}
BENCHMARK(BM_NormPDF);

static void BM_NormCDF(benchmark::State& state) {
    for (auto _ : state) {
        for (double x = -3.0; x <= 3.0; x += 0.1) {
            benchmark::DoNotOptimize(norm_cdf(x, 0.0, 1.0));
        }
    }
}
BENCHMARK(BM_NormCDF);

static void BM_NormPPF(benchmark::State& state) {
    for (auto _ : state) {
        for (double p = 0.01; p < 1.0; p += 0.01) {
            benchmark::DoNotOptimize(norm_ppf(p, 0.0, 1.0));
        }
    }
}
BENCHMARK(BM_NormPPF);

// ---------------------------------------------------------------------------
// t-distribution
// ---------------------------------------------------------------------------

static void BM_TPDF_DF5(benchmark::State& state) {
    for (auto _ : state) {
        for (double x = -5.0; x <= 5.0; x += 0.2) {
            benchmark::DoNotOptimize(t_pdf(x, 5.0));
        }
    }
}
BENCHMARK(BM_TPDF_DF5);

static void BM_TCDF_DF10(benchmark::State& state) {
    for (auto _ : state) {
        for (double x = -5.0; x <= 5.0; x += 0.2) {
            benchmark::DoNotOptimize(t_cdf(x, 10.0));
        }
    }
}
BENCHMARK(BM_TCDF_DF10);

// ---------------------------------------------------------------------------
// Exponential distribution
// ---------------------------------------------------------------------------

static void BM_ExpPDF(benchmark::State& state) {
    for (auto _ : state) {
        for (double x = 0.0; x <= 10.0; x += 0.2) {
            benchmark::DoNotOptimize(exp_pdf(x, 1.0));
        }
    }
}
BENCHMARK(BM_ExpPDF);

static void BM_ExpCDF(benchmark::State& state) {
    for (auto _ : state) {
        for (double x = 0.0; x <= 10.0; x += 0.2) {
            benchmark::DoNotOptimize(exp_cdf(x, 1.0));
        }
    }
}
BENCHMARK(BM_ExpCDF);

// ---------------------------------------------------------------------------
// Gamma distribution
// ---------------------------------------------------------------------------

static void BM_GammaPDF_Shape2(benchmark::State& state) {
    for (auto _ : state) {
        for (double x = 0.1; x <= 10.0; x += 0.2) {
            benchmark::DoNotOptimize(gamma_pdf(x, 2.0, 1.0));
        }
    }
}
BENCHMARK(BM_GammaPDF_Shape2);

static void BM_GammaPDF_Shape5(benchmark::State& state) {
    for (auto _ : state) {
        for (double x = 0.1; x <= 20.0; x += 0.2) {
            benchmark::DoNotOptimize(gamma_pdf(x, 5.0, 2.0));
        }
    }
}
BENCHMARK(BM_GammaPDF_Shape5);

// ---------------------------------------------------------------------------
// Binomial distribution
// ---------------------------------------------------------------------------

static void BM_BinomPDF_N20(benchmark::State& state) {
    for (auto _ : state) {
        for (int k = 0; k <= 20; ++k) {
            benchmark::DoNotOptimize(binom_pdf(k, 20, 0.4));
        }
    }
}
BENCHMARK(BM_BinomPDF_N20);

static void BM_BinomCDF_N50(benchmark::State& state) {
    for (auto _ : state) {
        for (int k = 0; k <= 50; ++k) {
            benchmark::DoNotOptimize(binom_cdf(k, 50, 0.3));
        }
    }
}
BENCHMARK(BM_BinomCDF_N50);

// ---------------------------------------------------------------------------
// Poisson distribution
// ---------------------------------------------------------------------------

static void BM_PoisPDF_Lambda5(benchmark::State& state) {
    for (auto _ : state) {
        for (int k = 0; k <= 30; ++k) {
            benchmark::DoNotOptimize(pois_pdf(static_cast<double>(k), 5.0));
        }
    }
}
BENCHMARK(BM_PoisPDF_Lambda5);

static void BM_PoisCDF_Lambda10(benchmark::State& state) {
    for (auto _ : state) {
        for (int k = 0; k <= 40; ++k) {
            benchmark::DoNotOptimize(pois_cdf(static_cast<double>(k), 10.0));
        }
    }
}
BENCHMARK(BM_PoisCDF_Lambda10);

// ---------------------------------------------------------------------------
// Chi-squared distribution
// ---------------------------------------------------------------------------

static void BM_Chi2PDF_DF5(benchmark::State& state) {
    for (auto _ : state) {
        for (double x = 0.1; x <= 20.0; x += 0.5) {
            benchmark::DoNotOptimize(chi2_pdf(x, 5.0));
        }
    }
}
BENCHMARK(BM_Chi2PDF_DF5);

static void BM_Chi2CDF_DF10(benchmark::State& state) {
    for (auto _ : state) {
        for (double x = 0.1; x <= 30.0; x += 0.5) {
            benchmark::DoNotOptimize(chi2_cdf(x, 10.0));
        }
    }
}
BENCHMARK(BM_Chi2CDF_DF10);

// ---------------------------------------------------------------------------
// Uniform distribution
// ---------------------------------------------------------------------------

static void BM_UniformPDF(benchmark::State& state) {
    for (auto _ : state) {
        for (double x = -0.5; x <= 1.5; x += 0.05) {
            benchmark::DoNotOptimize(uniform_pdf(x, 0.0, 1.0));
        }
    }
}
BENCHMARK(BM_UniformPDF);

static void BM_UniformCDF(benchmark::State& state) {
    for (auto _ : state) {
        for (double x = -0.5; x <= 1.5; x += 0.05) {
            benchmark::DoNotOptimize(uniform_cdf(x, 0.0, 1.0));
        }
    }
}
BENCHMARK(BM_UniformCDF);

BENCHMARK_MAIN();
