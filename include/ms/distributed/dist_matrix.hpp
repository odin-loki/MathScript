#pragma once

#include "ms/core/matrix.hpp"
#include "ms/distributed/mpi_context.hpp"
#include "ms/error/error_types.hpp"
#include <vector>

namespace ms::distributed {

enum class Distribution { Block, BlockCyclic };

template<typename S, template<typename> class Alloc = std::allocator>
struct DistMatrix {
    Matrix<S, StorageOrder::ColMajor, Alloc> local;
    Distribution distribution = Distribution::Block;
    int owner_rank = 0;
    size_t global_rows = 0;
    size_t global_cols = 0;
    std::vector<size_t> row_map;
};

template<typename S, template<typename> class Alloc = std::allocator>
Result<DistMatrix<S, Alloc>> scatter(
    const Matrix<S, StorageOrder::ColMajor, Alloc>& global,
    MPIContext& ctx,
    Distribution distribution = Distribution::Block);

template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> gather(
    const DistMatrix<S, Alloc>& dist,
    MPIContext& ctx);

template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> combine_gather(
    const std::vector<DistMatrix<S, Alloc>>& shards);

} // namespace ms::distributed
