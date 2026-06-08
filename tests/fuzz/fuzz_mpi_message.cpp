#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

#include "ms/core/matrix.hpp"
#include "ms/distributed/block.hpp"
#include "ms/distributed/dist_matrix.hpp"
#include "ms/distributed/mpi_context.hpp"

namespace {

double read_f64(const uint8_t* data, size_t size, size_t offset) {
    if (offset + sizeof(double) > size) {
        return 0.0;
    }
    double value = 0.0;
    std::memcpy(&value, data + offset, sizeof(double));
    if (!std::isfinite(value)) {
        return 0.0;
    }
    return value;
}

size_t bounded_dim(uint8_t byte, size_t max_dim) {
    return static_cast<size_t>(byte % static_cast<uint8_t>(max_dim + 1)) + 1;
}

ms::ColMatrix<double> make_matrix(const uint8_t* data, size_t size, size_t rows, size_t cols) {
    ms::ColMatrix<double> matrix(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const size_t offset = (2 + i * cols + j) % size;
            matrix(i, j) = read_f64(data, size, offset);
        }
    }
    return matrix;
}

std::vector<ms::distributed::DistMatrix<double>> simulate_shards(
    const ms::ColMatrix<double>& global,
    int nprocs,
    ms::distributed::Distribution distribution) {
    std::vector<ms::distributed::DistMatrix<double>> shards;
    shards.reserve(static_cast<size_t>(nprocs));
    for (int rank = 0; rank < nprocs; ++rank) {
        auto ctx = ms::distributed::init(0, nullptr);
        ctx.rank = rank;
        ctx.size = nprocs;
        shards.push_back(ms::distributed::scatter(global, ctx, distribution).value());
        ms::distributed::finalize(ctx);
    }
    return shards;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 24) {
        return 0;
    }

    const size_t rows = bounded_dim(data[0], 5);
    const size_t cols = bounded_dim(data[1], 4);
    const int nprocs = static_cast<int>(bounded_dim(data[2], 3));
    const bool block_cyclic = (data[3] & 1) != 0;
    const auto distribution =
        block_cyclic ? ms::distributed::Distribution::BlockCyclic : ms::distributed::Distribution::Block;

    auto ctx = ms::distributed::init(0, nullptr);
    (void)ms::distributed::backend_name(ctx);
    (void)ms::distributed::rank(ctx);
    (void)ms::distributed::size(ctx);

    const auto global = make_matrix(data, size, rows, cols);
    const auto dist = ms::distributed::scatter(global, ctx, distribution);
    if (dist) {
        const auto rebuilt = ms::distributed::gather(*dist, ctx);
        (void)rebuilt;
    }

    for (int rank = 0; rank < nprocs; ++rank) {
        const auto extent = ms::distributed::block_row_extent(rows, rank, nprocs);
        (void)extent.start;
        (void)extent.count;
        const auto cyclic = ms::distributed::block_cyclic_row_indices(rows, rank, nprocs);
        (void)cyclic.size();
    }

    const auto shards = simulate_shards(global, nprocs, distribution);
    (void)ms::distributed::combine_gather(shards);

    const double payload = read_f64(data, size, 16);
    (void)ms::distributed::allreduce_sum(ctx, payload);
    ms::distributed::barrier(ctx);
    ms::distributed::finalize(ctx);
    return 0;
}
