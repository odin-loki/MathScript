#include <benchmark/benchmark.h>
#include "ms/image/image.hpp"

namespace {

ms::image::Image make_slic_bench_rgb(int rows, int cols) {
    ms::image::Image img(rows, cols, 3, 0.f);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const float u = static_cast<float>(r) / static_cast<float>(rows);
            const float v = static_cast<float>(c) / static_cast<float>(cols);
            img.at(r, c, 0) = 0.2f + 0.6f * u;
            img.at(r, c, 1) = 0.1f + 0.5f * v;
            img.at(r, c, 2) = 0.3f + 0.4f * (u * v);
        }
    }
    return img;
}

} // namespace

static void BM_SLIC(benchmark::State& state) {
    const int rows = static_cast<int>(state.range(0));
    const int cols = static_cast<int>(state.range(1));
    const int superpixels = static_cast<int>(state.range(2));
    const auto rgb = make_slic_bench_rgb(rows, cols);

    for (auto _ : state) {
        benchmark::DoNotOptimize(ms::image::slic(rgb, superpixels, 10.0));
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * rows * cols);
}
BENCHMARK(BM_SLIC)->Args({128, 128, 64})->Args({256, 256, 256});
