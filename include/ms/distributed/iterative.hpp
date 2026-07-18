#pragma once

#include "ms/distributed/dist_matrix.hpp"

namespace ms::distributed {

template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_cg(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter = 1000,
    S tol = S(1e-10));

template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_gmres(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t restart = 20,
    size_t max_iter = 1000,
    S tol = S(1e-10));

template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_jacobi(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter = 1000,
    S tol = S(1e-10));

template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_bicgstab(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter = 1000,
    S tol = S(1e-10));

template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_minres(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter = 1000,
    S tol = S(1e-10));

template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_qmr(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter = 1000,
    S tol = S(1e-10));

template<typename S, template<typename> class Alloc = std::allocator>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_tfqmr(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter = 1000,
    S tol = S(1e-10));

} // namespace ms::distributed
