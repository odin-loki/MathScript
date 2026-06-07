// MathScript Core Operations Header

#pragma once

#include "ms/core/matrix.hpp"
#include "ms/error/error_types.hpp"
#include <cstdint>
#include <tuple>

namespace ms {

template<typename S, StorageOrder OA = StorageOrder::ColMajor, StorageOrder OB = StorageOrder::ColMajor,
         StorageOrder OC = StorageOrder::ColMajor,
         template<typename> class AllocA = std::allocator,
         template<typename> class AllocB = std::allocator,
         template<typename> class AllocC = std::allocator>
Result<Matrix<S, OC, AllocC>> matmul(
    const Matrix<S, OA, AllocA>& A,
    const Matrix<S, OB, AllocB>& B,
    int policy = 0);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<std::tuple<Matrix<S, OA, Alloc>, Matrix<S, OA, Alloc>, Matrix<S, OA, Alloc>>>
lu(const Matrix<S, OA, Alloc>& A);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<std::tuple<Matrix<S, OA, Alloc>, Matrix<S, OA, Alloc>>>
qr(const Matrix<S, OA, Alloc>& A);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>>
chol(const Matrix<S, OA, Alloc>& A);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>>
solve(const Matrix<S, OA, Alloc>& A, const Matrix<S, OA, Alloc>& b);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>>
expm(const Matrix<S, OA, Alloc>& A);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<S>
trace(const Matrix<S, OA, Alloc>& A);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<S>
det(const Matrix<S, OA, Alloc>& A);

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<S>
norm(const Matrix<S, OA, Alloc>& A, int p = 2);

template<typename S, template<typename> class Alloc = std::allocator>
Matrix<S, StorageOrder::ColMajor, Alloc>
eye(size_t n);

template<typename S, template<typename> class Alloc = std::allocator>
Matrix<S, StorageOrder::ColMajor, Alloc>
zeros(size_t m, size_t n);

template<typename S, template<typename> class Alloc = std::allocator>
Matrix<S, StorageOrder::ColMajor, Alloc>
ones(size_t m, size_t n);

template<typename S, StorageOrder Order, template<typename> class Alloc>
Matrix<S, StorageOrder::RowMajor, Alloc>
transpose(const Matrix<S, Order, Alloc>& A);

} // namespace ms
