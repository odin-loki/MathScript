// MathScript Benchmark: geo hot paths (point-in-polygon, convex hull, KD-tree)

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include "ms/geo/geo.hpp"

using ms::geo::KDTree2D;
using ms::geo::Point2D;
using ms::geo::Polygon2D;
using ms::geo::convex_hull_2d;
using ms::geo::point_in_polygon;
using ms::geo::signed_area;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

Polygon2D make_regular_polygon(int n, double radius) {
    Polygon2D poly;
    poly.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double theta = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n);
        poly.push_back({radius * std::cos(theta), radius * std::sin(theta)});
    }
    return poly;
}

std::vector<Point2D> make_random_points(int n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    std::vector<Point2D> pts;
    pts.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        pts.push_back({dist(rng), dist(rng)});
    return pts;
}

std::vector<Point2D> make_query_grid(int side, double span) {
    std::vector<Point2D> queries;
    queries.reserve(static_cast<size_t>(side) * static_cast<size_t>(side));
    const double step = span / static_cast<double>(side);
    for (int r = 0; r < side; ++r) {
        for (int c = 0; c < side; ++c) {
            queries.push_back({-span * 0.5 + step * (c + 0.5),
                               -span * 0.5 + step * (r + 0.5)});
        }
    }
    return queries;
}

} // namespace

static void BM_PointInPolygonBatch(benchmark::State& state) {
    const int verts = static_cast<int>(state.range(0));
    const int grid_side = static_cast<int>(state.range(1));
    const Polygon2D poly = make_regular_polygon(verts, 50.0);
    const std::vector<Point2D> queries = make_query_grid(grid_side, 120.0);
    for (auto _ : state) {
        int inside = 0;
        for (const auto& q : queries)
            inside += point_in_polygon(q, poly) ? 1 : 0;
        benchmark::DoNotOptimize(inside);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(queries.size()));
}
BENCHMARK(BM_PointInPolygonBatch)
    ->Args({64, 64})
    ->Args({128, 128})
    ->Args({256, 64});

static void BM_ConvexHull2D(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const std::vector<Point2D> pts = make_random_points(n, 226u);
    for (auto _ : state) {
        auto hull = convex_hull_2d(pts);
        benchmark::DoNotOptimize(hull);
        benchmark::DoNotOptimize(signed_area(hull));
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(n));
}
BENCHMARK(BM_ConvexHull2D)->Arg(256)->Arg(1024)->Arg(4096);

static void BM_KDTreeNearestBatch(benchmark::State& state) {
    const int n_pts = static_cast<int>(state.range(0));
    const int n_queries = static_cast<int>(state.range(1));
    const std::vector<Point2D> pts = make_random_points(n_pts, 42u);
    const KDTree2D tree(pts);
    const std::vector<Point2D> queries = make_random_points(n_queries, 99u);
    for (auto _ : state) {
        int sum = 0;
        for (const auto& q : queries)
            sum += tree.nearest(q);
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(n_queries));
}
BENCHMARK(BM_KDTreeNearestBatch)
    ->Args({10000, 1000})
    ->Args({50000, 1000})
    ->Args({100000, 500});
