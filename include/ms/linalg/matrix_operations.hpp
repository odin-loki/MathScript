// MathScript Matrix Operations Header

#pragma once

#include "ms/core/matrix.hpp"
#include "ms/error/expected.hpp"
#include "ms/memory/aligned_allocator.hpp"

namespace ms {

using Result = expected;

// Matrix multiply
template<typename S, StorageOrder OA, StorageOrder OB, StorageOrder OC, 
         template<typename> class AllocA, template<typename> class AllocB, template<typename> class AllocC>
Result<Matrix<S, OC, AllocC>> matmul(
    const Matrix<S, OA, AllocA>& A,
    const Matrix<S, OB, AllocB>& B,
    int policy = 0);

// LU decomposition
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<std::tuple<Matrix<S, OA, Alloc>, Matrix<S, OA, Alloc>, Matrix<int32_t, OA, memory::AlignedAllocator>>>
lu(const Matrix<S, OA, Alloc>& A);

// QR decomposition
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<std::tuple<Matrix<S, OA, Alloc>, Matrix<S, OA, Alloc>>>
qr(const Matrix<S, OA, Alloc>& A);

// Cholesky decomposition
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>>
chol(const Matrix<S, OA, Alloc>& A);

// Solve linear system
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>>
solve(const Matrix<S, OA, Alloc>& A, const Matrix<S, OA, Alloc>& b);

// Matrix exponential
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>>
expm(const Matrix<S, OA, Alloc>& A);

// Matrix trace
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<S>
trace(const Matrix<S, OA, Alloc>& A);

// Matrix determinant
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<S>
det(const Matrix<S, OA, Alloc>& A);

// Matrix norm
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<S>
norm(const Matrix<S, OA, Alloc>& A, int p = 2);

// Eye matrix
template<typename S, template<typename> class Alloc>
Matrix<S, StorageOrder::ColMajor, Alloc>
eye(size_t n);

// Zeros matrix
template<typename S, template<typename> class Alloc>
Matrix<S, StorageOrder::ColMajor, Alloc>
zeros(size_t m, size_t n);

// Ones matrix
template<typename S, template<typename> class Alloc>
Matrix<S, StorageOrder::ColMajor, Alloc>
ones(size_t m, size_t n);

// Transpose
template<typename S, StorageOrder Order, template<typename> class Alloc>
Matrix<S, StorageOrder::RowMajor, memory::AlignedAllocator>
transpose(const Matrix<S, Order, Alloc>& A);

} // namespace ms