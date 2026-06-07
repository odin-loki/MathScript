#pragma once

#include "ms/distributed/dist_matrix.hpp"

namespace ms::distributed {

template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> solve(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx);

} // namespace ms::distributed
