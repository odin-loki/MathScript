// MathScript: Benchmarks for global optimizers and ML ensemble/clustering methods

#include <benchmark/benchmark.h>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

#include "ms/ml/ml.hpp"
#include "ms/optim/optim.hpp"

using namespace ms;
using ms::ml::GaussianMixture;
using ms::ml::GradientBoosting;
using ms::ml::Mat;
using ms::ml::RandomForest;
using ms::ml::SVM;
using ms::ml::SVMKernel;
using ms::ml::Vec;

// ---------------------------------------------------------------------------
// Optimization: 2D sphere f(x) = x[0]^2 + x[1]^2
// ---------------------------------------------------------------------------

static double sphere2d(const std::vector<double>& x) {
    return x[0] * x[0] + x[1] * x[1];
}

static void BM_SimulatedAnnealing(benchmark::State& state) {
    auto f = [](const std::vector<double>& x) { return sphere2d(x); };
    for (auto _ : state) {
        OptimResult r = simulated_annealing(f, {2.0, 2.0}, 1.0, 0.99, 2000, 42);
        benchmark::DoNotOptimize(r.f_val);
    }
}
BENCHMARK(BM_SimulatedAnnealing);

static void BM_DifferentialEvolution(benchmark::State& state) {
    auto f = [](const std::vector<double>& x) { return sphere2d(x); };
    std::vector<std::pair<double, double>> bounds = {{-5.0, 5.0}, {-5.0, 5.0}};
    for (auto _ : state) {
        OptimResult r = differential_evolution(f, bounds, 20, 0.8, 0.9, 500, 42);
        benchmark::DoNotOptimize(r.f_val);
    }
}
BENCHMARK(BM_DifferentialEvolution);

static void BM_ParticleSwarm(benchmark::State& state) {
    auto f = [](const std::vector<double>& x) { return sphere2d(x); };
    std::vector<std::pair<double, double>> bounds = {{-5.0, 5.0}, {-5.0, 5.0}};
    for (auto _ : state) {
        OptimResult r = particle_swarm(f, bounds, 20, 200, 42);
        benchmark::DoNotOptimize(r.f_val);
    }
}
BENCHMARK(BM_ParticleSwarm);

static void BM_CMAES(benchmark::State& state) {
    auto f = [](const std::vector<double>& x) { return sphere2d(x); };
    for (auto _ : state) {
        OptimResult r = cmaes(f, {2.0, 3.0}, 0.5, 500, 42);
        benchmark::DoNotOptimize(r.f_val);
    }
}
BENCHMARK(BM_CMAES);

// ---------------------------------------------------------------------------
// ML: synthetic datasets (generated once outside the timing loop)
// ---------------------------------------------------------------------------

static std::pair<Mat, Vec> make_classification_data() {
    constexpr int n_rows = 300;
    constexpr int n_features = 5;
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    Mat X;
    Vec y;
    X.reserve(n_rows);
    y.reserve(n_rows);
    for (int i = 0; i < n_rows; ++i) {
        Vec row(n_features);
        const double shift = (i < n_rows / 2) ? -1.0 : 1.0;
        for (int j = 0; j < n_features; ++j)
            row[j] = dist(rng) + shift;
        X.push_back(std::move(row));
        y.push_back(i < n_rows / 2 ? 0.0 : 1.0);
    }
    return {X, y};
}

static std::pair<Mat, Vec> make_regression_data() {
    constexpr int n_rows = 300;
    constexpr int n_features = 4;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Mat X;
    Vec y;
    X.reserve(n_rows);
    y.reserve(n_rows);
    for (int i = 0; i < n_rows; ++i) {
        Vec row(n_features);
        double target = 0.0;
        for (int j = 0; j < n_features; ++j) {
            row[j] = dist(rng);
            target += (j + 1) * row[j];
        }
        X.push_back(std::move(row));
        y.push_back(target + 0.1 * dist(rng));
    }
    return {X, y};
}

static std::pair<Mat, Vec> make_svm_data() {
    constexpr int n_rows = 200;
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 0.5);

    Mat X;
    Vec y;
    X.reserve(n_rows);
    y.reserve(n_rows);
    for (int i = 0; i < n_rows; ++i) {
        const double x0 = (i < n_rows / 2) ? -1.5 - 0.05 * i : 1.5 + 0.05 * (i - n_rows / 2);
        const double x1 = dist(rng);
        X.push_back({x0, x1});
        y.push_back(x0 < 0.0 ? -1.0 : 1.0);
    }
    return {X, y};
}

static Mat make_gmm_data() {
    constexpr int n_rows = 300;
    constexpr int n_features = 3;
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 0.3);

    Mat X;
    X.reserve(n_rows);
    for (int i = 0; i < n_rows; ++i) {
        const int cluster = i / (n_rows / 3);
        Vec row(n_features);
        for (int j = 0; j < n_features; ++j)
            row[j] = static_cast<double>(cluster * 5 + j) + dist(rng);
        X.push_back(std::move(row));
    }
    return X;
}

static void BM_RandomForest(benchmark::State& state) {
    const auto [X, y] = make_classification_data();
    for (auto _ : state) {
        RandomForest rf;
        rf.config.n_trees = 25;
        rf.config.max_depth = 5;
        rf.config.seed = 42;
        rf.fit(X, y);
        benchmark::DoNotOptimize(&rf.trees);
    }
}
BENCHMARK(BM_RandomForest);

static void BM_GradientBoosting(benchmark::State& state) {
    const auto [X, y] = make_regression_data();
    for (auto _ : state) {
        GradientBoosting gb;
        gb.config.n_trees = 40;
        gb.config.max_depth = 3;
        gb.config.learning_rate = 0.1;
        gb.config.seed = 42;
        gb.fit(X, y);
        benchmark::DoNotOptimize(&gb.trees);
    }
}
BENCHMARK(BM_GradientBoosting);

static void BM_SVM(benchmark::State& state) {
    const auto [X, y] = make_svm_data();
    for (auto _ : state) {
        SVM svm;
        svm.config.kernel = SVMKernel::Linear;
        svm.config.C = 10.0;
        svm.config.max_iter = 500;
        svm.fit(X, y);
        benchmark::DoNotOptimize(&svm.support_vectors);
    }
}
BENCHMARK(BM_SVM);

static void BM_GaussianMixture(benchmark::State& state) {
    const Mat X = make_gmm_data();
    for (auto _ : state) {
        GaussianMixture gmm;
        gmm.config.n_components = 3;
        gmm.config.max_iter = 50;
        gmm.config.seed = 42;
        gmm.fit(X);
        benchmark::DoNotOptimize(&gmm.means);
    }
}
BENCHMARK(BM_GaussianMixture);

BENCHMARK_MAIN();
