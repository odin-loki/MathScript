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

ms::image::Image make_watershed_bench_gray(int rows, int cols) {
    ms::image::Image gray(rows, cols, 1, 0.f);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const float d1 = std::hypot(static_cast<float>(r - rows / 2),
                                          static_cast<float>(c - cols / 4));
            const float d2 = std::hypot(static_cast<float>(r - rows / 2),
                                          static_cast<float>(c - 3 * cols / 4));
            gray.at(r, c, 0) = std::max(0.f, 1.f - d1 / 8.f) + std::max(0.f, 1.f - d2 / 8.f);
        }
    }
    return gray;
}

ms::image::Image make_watershed_bench_markers(int rows, int cols) {
    ms::image::Image markers(rows, cols, 1, 0.f);
    markers.at(rows / 2, cols / 4, 0) = 1.f;
    markers.at(rows / 2, 3 * cols / 4, 0) = 2.f;
    return markers;
}

ms::image::Image make_gaussian_bench_image(int rows, int cols, int channels) {
    ms::image::Image img(rows, cols, channels, 0.f);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const float u = static_cast<float>(r) / static_cast<float>(rows);
            const float v = static_cast<float>(c) / static_cast<float>(cols);
            for (int ch = 0; ch < channels; ++ch) {
                img.at(r, c, ch) = 0.15f + 0.7f * (u * 0.6f + v * 0.4f) +
                                   0.05f * static_cast<float>(ch);
            }
        }
    }
    return img;
}

} // namespace

static void BM_GaussianBlur(benchmark::State& state) {
    const int rows = static_cast<int>(state.range(0));
    const int cols = static_cast<int>(state.range(1));
    const int channels = static_cast<int>(state.range(2));
    const float sigma = static_cast<float>(state.range(3)) * 0.1f;
    const auto img = make_gaussian_bench_image(rows, cols, channels);

    for (auto _ : state) {
        benchmark::DoNotOptimize(ms::image::imgaussfilt(img, sigma));
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * rows * cols * channels);
}
BENCHMARK(BM_GaussianBlur)
    ->Args({256, 256, 1, 10})
    ->Args({512, 512, 1, 15})
    ->Args({256, 256, 3, 10});

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

static void BM_Watershed(benchmark::State& state) {
    const int rows = static_cast<int>(state.range(0));
    const int cols = static_cast<int>(state.range(1));
    const auto gray = make_watershed_bench_gray(rows, cols);
    const auto markers = make_watershed_bench_markers(rows, cols);

    for (auto _ : state) {
        benchmark::DoNotOptimize(ms::image::watershed(gray, markers));
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * rows * cols);
}
BENCHMARK(BM_Watershed)->Args({64, 64})->Args({128, 128})->Args({256, 256});

BENCHMARK_MAIN();
