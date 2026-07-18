// MathScript Benchmarks: Distributed and CellAI/Cypha
// Benchmarks for MPIContext operations, DistMatrix scatter/gather,
// distributed solve, CellMemory step/recall, DifModel update/predict

#include <benchmark/benchmark.h>
#include <array>
#include <vector>

#include "ms/distributed/dist_matrix.hpp"
#include "ms/distributed/iterative.hpp"
#include "ms/distributed/mpi_context.hpp"
#include "ms/distributed/solve.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/cypha/cypha.hpp"
#include "ms/core/matrix.hpp"
#include "ms/cuda/nccl.hpp"

using namespace ms;
using namespace ms::distributed;
using DMatrix = ms::Matrix<double>;

// ---------------------------------------------------------------------------
// Distributed: MPIContext init/finalize
// ---------------------------------------------------------------------------

static void BM_MPIContext_Init(benchmark::State& state) {
    for (auto _ : state) {
        auto ctx = init(0, nullptr);
        benchmark::DoNotOptimize(ctx);
        finalize(ctx);
    }
}
BENCHMARK(BM_MPIContext_Init);

static void BM_MPIContext_AllreduceSum(benchmark::State& state) {
    auto ctx = init(0, nullptr);
    for (auto _ : state) {
        double result = allreduce_sum(ctx, 42.0);
        benchmark::DoNotOptimize(result);
    }
    finalize(ctx);
}
BENCHMARK(BM_MPIContext_AllreduceSum);

static void BM_CudaAllreduceAvg(benchmark::State& state) {
    for (auto _ : state) {
        const double result = ms::cuda::allreduce_avg(2.0);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_CudaAllreduceAvg);

static void BM_MPIContext_Barrier(benchmark::State& state) {
    auto ctx = init(0, nullptr);
    for (auto _ : state) {
        barrier(ctx);
    }
    finalize(ctx);
}
BENCHMARK(BM_MPIContext_Barrier);

// ---------------------------------------------------------------------------
// Distributed: scatter/gather 4x4
// ---------------------------------------------------------------------------

static void BM_DistMatrix_Scatter_4x4(benchmark::State& state) {
    auto ctx = init(0, nullptr);
    DMatrix A(4, 4, 1.0);
    for (auto _ : state) {
        auto dm = scatter(A, ctx);
        benchmark::DoNotOptimize(dm);
    }
    finalize(ctx);
}
BENCHMARK(BM_DistMatrix_Scatter_4x4);

static void BM_DistMatrix_Scatter_64x64(benchmark::State& state) {
    auto ctx = init(0, nullptr);
    DMatrix A(64, 64, 1.0);
    for (auto _ : state) {
        auto dm = scatter(A, ctx);
        benchmark::DoNotOptimize(dm);
    }
    finalize(ctx);
}
BENCHMARK(BM_DistMatrix_Scatter_64x64);

static void BM_DistMatrix_ScatterGather_4x4(benchmark::State& state) {
    auto ctx = init(0, nullptr);
    DMatrix A(4, 4, 2.0);
    for (auto _ : state) {
        auto dm = scatter(A, ctx);
        if (dm.has_value()) {
            auto gathered = gather(dm.value(), ctx);
            benchmark::DoNotOptimize(gathered);
        }
    }
    finalize(ctx);
}
BENCHMARK(BM_DistMatrix_ScatterGather_4x4);

static void BM_DistMatrix_ScatterGather_64x64(benchmark::State& state) {
    auto ctx = init(0, nullptr);
    DMatrix A(64, 64, 2.0);
    for (auto _ : state) {
        auto dm = scatter(A, ctx);
        if (dm.has_value()) {
            auto gathered = gather(dm.value(), ctx);
            benchmark::DoNotOptimize(gathered);
        }
    }
    finalize(ctx);
}
BENCHMARK(BM_DistMatrix_ScatterGather_64x64);

static void BM_DistMatrix_ScatterGather_BlockCyclic_64x64(benchmark::State& state) {
    auto ctx = init(0, nullptr);
    DMatrix A(64, 64, 3.0);
    for (auto _ : state) {
        auto dm = scatter(A, ctx, Distribution::BlockCyclic);
        if (dm.has_value()) {
            auto gathered = gather(dm.value(), ctx);
            benchmark::DoNotOptimize(gathered);
        }
    }
    finalize(ctx);
}
BENCHMARK(BM_DistMatrix_ScatterGather_BlockCyclic_64x64);

// ---------------------------------------------------------------------------
// Distributed: solve 2x2 system
// ---------------------------------------------------------------------------

static void BM_DistributedSolve_2x2(benchmark::State& state) {
    auto ctx = init(0, nullptr);
    DMatrix A({{4.0, 1.0}, {1.0, 3.0}});
    DMatrix b({{1.0}, {2.0}});

    for (auto _ : state) {
        state.PauseTiming();
        auto dA = scatter(A, ctx).value();
        auto db = scatter(b, ctx).value();
        state.ResumeTiming();

        auto result = distributed::solve(dA, db, ctx);
        benchmark::DoNotOptimize(result);
    }
    finalize(ctx);
}
BENCHMARK(BM_DistributedSolve_2x2);

static void BM_DistCg_2x2(benchmark::State& state) {
    auto ctx = init(0, nullptr);
    DMatrix A({{4.0, 1.0}, {1.0, 3.0}});
    DMatrix b({{1.0}, {2.0}});

    for (auto _ : state) {
        state.PauseTiming();
        auto dA = scatter(A, ctx).value();
        auto db = scatter(b, ctx).value();
        state.ResumeTiming();

        auto result = dist_cg(dA, db, ctx);
        benchmark::DoNotOptimize(result);
    }
    finalize(ctx);
}
BENCHMARK(BM_DistCg_2x2);

static void BM_DistGmres_2x2(benchmark::State& state) {
    auto ctx = init(0, nullptr);
    DMatrix A({{4.0, 1.0}, {1.0, 3.0}});
    DMatrix b({{1.0}, {2.0}});

    for (auto _ : state) {
        state.PauseTiming();
        auto dA = scatter(A, ctx).value();
        auto db = scatter(b, ctx).value();
        state.ResumeTiming();

        auto result = dist_gmres(dA, db, ctx);
        benchmark::DoNotOptimize(result);
    }
    finalize(ctx);
}
BENCHMARK(BM_DistGmres_2x2);

static void BM_DistJacobi_2x2(benchmark::State& state) {
    auto ctx = init(0, nullptr);
    DMatrix A({{4.0, 1.0}, {1.0, 3.0}});
    DMatrix b({{1.0}, {2.0}});

    for (auto _ : state) {
        state.PauseTiming();
        auto dA = scatter(A, ctx).value();
        auto db = scatter(b, ctx).value();
        state.ResumeTiming();

        auto result = dist_jacobi(dA, db, ctx);
        benchmark::DoNotOptimize(result);
    }
    finalize(ctx);
}
BENCHMARK(BM_DistJacobi_2x2);

static void BM_DistBicgstab_2x2(benchmark::State& state) {
    auto ctx = init(0, nullptr);
    DMatrix A({{3.0, 1.0}, {1.0, 2.0}});
    DMatrix b({{1.0}, {1.0}});

    for (auto _ : state) {
        state.PauseTiming();
        auto dA = scatter(A, ctx).value();
        auto db = scatter(b, ctx).value();
        state.ResumeTiming();

        auto result = dist_bicgstab(dA, db, ctx);
        benchmark::DoNotOptimize(result);
    }
    finalize(ctx);
}
BENCHMARK(BM_DistBicgstab_2x2);

static void BM_DistMinres_2x2(benchmark::State& state) {
    auto ctx = init(0, nullptr);
    DMatrix A({{4.0, 1.0}, {1.0, 3.0}});
    DMatrix b({{1.0}, {2.0}});

    for (auto _ : state) {
        state.PauseTiming();
        auto dA = scatter(A, ctx).value();
        auto db = scatter(b, ctx).value();
        state.ResumeTiming();

        auto result = dist_minres(dA, db, ctx);
        benchmark::DoNotOptimize(result);
    }
    finalize(ctx);
}
BENCHMARK(BM_DistMinres_2x2);

static void BM_DistQmr_2x2(benchmark::State& state) {
    auto ctx = init(0, nullptr);
    DMatrix A({{4.0, 1.0}, {1.0, 3.0}});
    DMatrix b({{1.0}, {2.0}});

    for (auto _ : state) {
        state.PauseTiming();
        auto dA = scatter(A, ctx).value();
        auto db = scatter(b, ctx).value();
        state.ResumeTiming();

        auto result = dist_qmr(dA, db, ctx);
        benchmark::DoNotOptimize(result);
    }
    finalize(ctx);
}
BENCHMARK(BM_DistQmr_2x2);

static void BM_DistTfqmr_2x2(benchmark::State& state) {
    auto ctx = init(0, nullptr);
    DMatrix A({{4.0, 1.0}, {1.0, 3.0}});
    DMatrix b({{1.0}, {2.0}});

    for (auto _ : state) {
        state.PauseTiming();
        auto dA = scatter(A, ctx).value();
        auto db = scatter(b, ctx).value();
        state.ResumeTiming();

        auto result = dist_tfqmr(dA, db, ctx);
        benchmark::DoNotOptimize(result);
    }
    finalize(ctx);
}
BENCHMARK(BM_DistTfqmr_2x2);

static void BM_DistLsmr_2x2(benchmark::State& state) {
    auto ctx = init(0, nullptr);
    DMatrix A({{4.0, 1.0}, {1.0, 3.0}});
    DMatrix b({{1.0}, {2.0}});

    for (auto _ : state) {
        state.PauseTiming();
        auto dA = scatter(A, ctx).value();
        auto db = scatter(b, ctx).value();
        state.ResumeTiming();

        auto result = dist_lsmr(dA, db, ctx);
        benchmark::DoNotOptimize(result);
    }
    finalize(ctx);
}
BENCHMARK(BM_DistLsmr_2x2);

// ---------------------------------------------------------------------------
// CellAI: CellMemory::step
// ---------------------------------------------------------------------------

static void BM_CellMemory_Step_4x8(benchmark::State& state) {
    ms::cellai::CellMemory memory(4, 8, {0.1, 1.0, 10.0});
    DMatrix x(4, 1, 0.5);
    for (auto _ : state) {
        auto result = memory.step(x);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_CellMemory_Step_4x8);

static void BM_CellMemory_Step_16x32(benchmark::State& state) {
    ms::cellai::CellMemory memory(16, 32, {0.1, 1.0, 10.0, 100.0});
    DMatrix x(16, 1, 0.1);
    for (auto _ : state) {
        auto result = memory.step(x);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_CellMemory_Step_16x32);

static void BM_CellMemory_Recall(benchmark::State& state) {
    ms::cellai::CellMemory memory(8, 16, {0.5, 5.0});
    DMatrix x(8, 1, 1.0);
    memory.step(x);
    memory.step(x);
    for (auto _ : state) {
        auto result = memory.recall(0.5);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_CellMemory_Recall);

static void BM_HebbianUpdate_4x4(benchmark::State& state) {
    DMatrix w(4, 4, 0.1);
    DMatrix x(4, 1, 0.5);
    DMatrix y(4, 1, 0.5);
    for (auto _ : state) {
        auto w_new = ms::cellai::hebbian_update(w, x, y, 0.01);
        benchmark::DoNotOptimize(w_new);
    }
}
BENCHMARK(BM_HebbianUpdate_4x4);

static void BM_CellToCyphaFeatures(benchmark::State& state) {
    const std::vector<double> scales = {0.1, 1.0, 10.0};
    ms::cellai::CellMemory memory(4, 8, scales);
    DMatrix x(4, 1, 0.3);
    memory.step(x);
    for (auto _ : state) {
        auto f = ms::cellai::cell_to_cypha_features(memory, scales);
        benchmark::DoNotOptimize(f);
    }
}
BENCHMARK(BM_CellToCyphaFeatures);

// ---------------------------------------------------------------------------
// Cypha: DifModel update/predict/ood_score
// ---------------------------------------------------------------------------

static void BM_DifModel_Update_2x1(benchmark::State& state) {
    ms::cypha::DifConfig cfg;
    cfg.input_dim = 2;
    cfg.output_dim = 1;
    cfg.n_experts = 4;
    ms::cypha::DifModel model(cfg);
    DMatrix x(2, 1, 0.5);
    DMatrix y(1, 1, 1.0);
    for (auto _ : state) {
        auto r = model.update(x, y);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_DifModel_Update_2x1);

static void BM_DifModel_Predict_2x1(benchmark::State& state) {
    ms::cypha::DifConfig cfg;
    cfg.input_dim = 2;
    cfg.output_dim = 1;
    cfg.n_experts = 4;
    ms::cypha::DifModel model(cfg);
    DMatrix x(2, 1, 0.5);
    DMatrix y(1, 1, 1.0);
    model.update(x, y);
    model.update(x, y);
    for (auto _ : state) {
        auto r = model.predict(x);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_DifModel_Predict_2x1);

static void BM_DifModel_OOD_Score(benchmark::State& state) {
    ms::cypha::DifConfig cfg;
    cfg.input_dim = 4;
    cfg.output_dim = 1;
    cfg.n_experts = 8;
    ms::cypha::DifModel model(cfg);
    DMatrix x(4, 1, 0.0);
    DMatrix y(1, 1, 0.0);
    for (int i = 0; i < 5; ++i) model.update(x, y);
    for (auto _ : state) {
        auto r = model.ood_score(x);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_DifModel_OOD_Score);

static void BM_NIGFit(benchmark::State& state) {
    DMatrix data(20, 1, 0.0);
    for (int i = 0; i < 20; ++i) data(i, 0) = static_cast<double>(i) * 0.05 - 0.5;
    for (auto _ : state) {
        auto params = ms::cypha::nig_fit(data);
        benchmark::DoNotOptimize(params);
    }
}
BENCHMARK(BM_NIGFit);

static void BM_NIGPdf(benchmark::State& state) {
    ms::cypha::NIGParams params;
    params.mu = 0.0;
    params.alpha = 2.0;
    params.beta = 0.0;
    params.delta = 1.0;
    for (auto _ : state) {
        double v = ms::cypha::nig_pdf(0.5, params);
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_NIGPdf);

BENCHMARK_MAIN();
