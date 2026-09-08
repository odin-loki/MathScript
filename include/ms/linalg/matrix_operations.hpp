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

/// @brief Matrix exponential e^A.
///
/// @note  Scaling and squaring with a Pade approximant (Higham 2005): the
///        order m is the smallest of 3/5/7/9/13 whose backward-error bound
///        theta_m covers ||A||_1; above theta_13 the matrix is scaled by 2^-s,
///        the order-13 approximant is evaluated, and the result is squared s
///        times. There is no norm restriction and no accuracy caveat -- the
///        relative error is at round-off level for any ||A||. A low-precision
///        scalar type (float) caps the ladder at m = 7, whose coefficients fit
///        a 24-bit significand, and scales further instead.
/// @return DimensionMismatch if A is not square; DomainError if A has
///         non-finite entries; SingularMatrix in the (pathological) case where
///         the Pade denominator cannot be factorised.
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

/// @brief Matrix / vector norm selected by @p p.
///
/// @note  The supported values, and exactly what each computes:
///        - p =  2 (default): the ENTRYWISE 2-norm, i.e. the Frobenius norm
///          sqrt(sum a_ij^2). For a vector this is the usual Euclidean norm.
///          It is NOT the induced spectral norm sigma_max — use svd() for that.
///        - p =  1: the ENTRYWISE 1-norm, sum |a_ij|. For a vector this is the
///          usual 1-norm; for a matrix it is not the induced max-column-sum.
///        - p >  2: the entrywise p-norm (sum |a_ij|^p)^(1/p).
///        - p = -1: the INDUCED infinity norm, the largest absolute row sum.
///          For a column vector this is max |v_i|.
///        - p =  0: the l0 count of nonzero entries (as used in sparse
///          recovery); not a norm, but a well-defined quantity.
///        Every other p (that is, p <= -2) is a DomainError rather than a
///        silently returned zero.
/// @note  p = 2 and p > 2 scale by the largest entry before accumulating, so
///        they do not overflow or underflow on extreme inputs.
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