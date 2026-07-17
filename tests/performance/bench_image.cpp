#include <benchmark/benchmark.h>
#include <cmath>
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

static void BM_Boxfilter(benchmark::State& state) {
    const int rows = static_cast<int>(state.range(0));
    const int cols = static_cast<int>(state.range(1));
    const int channels = static_cast<int>(state.range(2));
    const int ksize = static_cast<int>(state.range(3));
    const auto img = make_gaussian_bench_image(rows, cols, channels);

    for (auto _ : state) {
        benchmark::DoNotOptimize(ms::image::boxfilter(img, ksize));
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * rows * cols * channels);
}
BENCHMARK(BM_Boxfilter)
    ->Args({256, 256, 1, 15})
    ->Args({512, 512, 1, 31})
    ->Args({256, 256, 3, 15});

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
BENCHMARK(BM_SLIC)->Args({128, 128, 64})->Args({192, 192, 96});

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

static void BM_Medfilt2(benchmark::State& state) {
    const int rows = static_cast<int>(state.range(0));
    const int cols = static_cast<int>(state.range(1));
    const int ksize = static_cast<int>(state.range(2));
    const auto img = make_gaussian_bench_image(rows, cols, 1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(ms::image::medfilt2(img, ksize));
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * rows * cols);
}
BENCHMARK(BM_Medfilt2)
    ->Args({256, 256, 3})
    ->Args({512, 512, 3})
    ->Args({256, 256, 5})
    ->Args({512, 512, 5})
    ->Args({256, 256, 7});

static void BM_Imresize(benchmark::State& state) {
    const int rows = static_cast<int>(state.range(0));
    const int cols = static_cast<int>(state.range(1));
    const int channels = static_cast<int>(state.range(2));
    const int out_rows = static_cast<int>(state.range(3));
    const int out_cols = static_cast<int>(state.range(4));
    const auto img = make_gaussian_bench_image(rows, cols, channels);

    for (auto _ : state) {
        benchmark::DoNotOptimize(ms::image::imresize(img, out_rows, out_cols));
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * out_rows * out_cols * channels);
}
BENCHMARK(BM_Imresize)
    ->Args({256, 256, 1, 128, 128})
    ->Args({512, 512, 3, 256, 256})
    ->Args({256, 256, 3, 512, 512});

static void BM_Bilateral(benchmark::State& state) {
    const int rows = static_cast<int>(state.range(0));
    const int cols = static_cast<int>(state.range(1));
    const int channels = static_cast<int>(state.range(2));
    const float sigma_s = static_cast<float>(state.range(3)) * 0.1f;
    const float sigma_r = static_cast<float>(state.range(4)) * 0.01f;
    const auto img = make_gaussian_bench_image(rows, cols, channels);

    for (auto _ : state) {
        benchmark::DoNotOptimize(ms::image::bilateral(img, sigma_s, sigma_r));
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * rows * cols * channels);
}
BENCHMARK(BM_Bilateral)
    ->Args({128, 128, 1, 10, 10})
    ->Args({256, 256, 1, 10, 10})
    ->Args({128, 128, 3, 10, 10});

static std::vector<std::vector<float>> make_sobel_x_kernel() {
    return {{-1.f, 0.f, 1.f}, {-2.f, 0.f, 2.f}, {-1.f, 0.f, 1.f}};
}

static std::vector<std::vector<float>> make_separable_gaussian_kernel(int ksize) {
    const int half = ksize / 2;
    std::vector<float> g1d(static_cast<std::size_t>(ksize));
    float sum = 0.f;
    for (int i = 0; i < ksize; ++i) {
        const float x = static_cast<float>(i - half);
        g1d[static_cast<std::size_t>(i)] = std::exp(-x * x / 8.f);
        sum += g1d[static_cast<std::size_t>(i)];
    }
    const float inv_sum = 1.f / sum;
    for (auto& v : g1d) {
        v *= inv_sum;
    }

    std::vector<std::vector<float>> K(
        static_cast<std::size_t>(ksize),
        std::vector<float>(static_cast<std::size_t>(ksize)));
    for (int r = 0; r < ksize; ++r) {
        for (int c = 0; c < ksize; ++c) {
            K[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                g1d[static_cast<std::size_t>(r)] * g1d[static_cast<std::size_t>(c)];
        }
    }
    return K;
}

static void BM_Imfilter(benchmark::State& state) {
    const int rows = static_cast<int>(state.range(0));
    const int cols = static_cast<int>(state.range(1));
    const int channels = static_cast<int>(state.range(2));
    const int kernel_kind = static_cast<int>(state.range(3));
    const auto img = make_gaussian_bench_image(rows, cols, channels);
    const auto k3 = make_sobel_x_kernel();
    const auto k_sep = make_separable_gaussian_kernel(15);

    for (auto _ : state) {
        if (kernel_kind == 0) {
            benchmark::DoNotOptimize(ms::image::imfilter(img, k3));
        } else {
            benchmark::DoNotOptimize(ms::image::imfilter(img, k_sep));
        }
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * rows * cols * channels);
}
BENCHMARK(BM_Imfilter)
    ->Args({256, 256, 1, 0})
    ->Args({512, 512, 1, 0})
    ->Args({256, 256, 3, 0})
    ->Args({256, 256, 1, 1})
    ->Args({512, 512, 1, 1});

static void BM_Sharpen(benchmark::State& state) {
    const int rows = static_cast<int>(state.range(0));
    const int cols = static_cast<int>(state.range(1));
    const int channels = static_cast<int>(state.range(2));
    const auto img = make_gaussian_bench_image(rows, cols, channels);

    for (auto _ : state) {
        benchmark::DoNotOptimize(ms::image::sharpen(img));
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * rows * cols * channels);
}
BENCHMARK(BM_Sharpen)
    ->Args({256, 256, 1})
    ->Args({512, 512, 1})
    ->Args({256, 256, 3});

BENCHMARK_MAIN();
