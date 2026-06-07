#pragma once

#include "ms/distributed/dist_matrix.hpp"
#include "ms/distributed/mpi_context.hpp"
#include "ms/linalg/linalg.hpp"
#include <tuple>

namespace ms::distributed {

template<typename S, template<typename> class Alloc = std::allocator>
Result<EigResult> eig_sym(const DistMatrix<S, Alloc>& A, MPIContext& ctx) {
    auto global = gather(A, ctx);
    if (!global) {
        return std::unexpected(global.error());
    }
    if (rank(ctx) != 0 && size(ctx) > 1) {
        return EigResult{};
    }
    return ms::eig_sym(*global);
}

template<typename S, template<typename> class Alloc = std::allocator>
Result<SvdResult> svd(const DistMatrix<S, Alloc>& A, MPIContext& ctx) {
    auto global = gather(A, ctx);
    if (!global) {
        return std::unexpected(global.error());
    }
    if (rank(ctx) != 0 && size(ctx) > 1) {
        return SvdResult{};
    }
    return ms::svd(*global);
}

template<typename S, template<typename> class Alloc = std::allocator>
Result<std::tuple<
    Matrix<S, StorageOrder::ColMajor, Alloc>,
    Matrix<S, StorageOrder::ColMajor, Alloc>,
    Matrix<S, StorageOrder::ColMajor, Alloc>>> lu(
    const DistMatrix<S, Alloc>& A,
    MPIContext& ctx) {
    auto global = gather(A, ctx);
    if (!global) {
        return std::unexpected(global.error());
    }
    if (rank(ctx) != 0 && size(ctx) > 1) {
        return std::tuple<
            Matrix<S, StorageOrder::ColMajor, Alloc>,
            Matrix<S, StorageOrder::ColMajor, Alloc>,
            Matrix<S, StorageOrder::ColMajor, Alloc>>{};
    }
    return ms::lu(*global);
}

} // namespace ms::distributed
