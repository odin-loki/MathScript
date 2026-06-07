#pragma once

#include "ms/core/matrix.hpp"
#include <cmath>
#include <limits>
#include <vector>

namespace ms::linalg_detail {

template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, OA, Alloc> copy(const Matrix<S, OA, Alloc>& A) {
    Matrix<S, OA, Alloc> B(A.rows(), A.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            B(i, j) = A(i, j);
        }
    }
    return B;
}

template<typename S, StorageOrder OA, StorageOrder OB, StorageOrder OC,
         template<typename> class AllocA, template<typename> class AllocB, template<typename> class AllocC>
Matrix<S, OC, AllocC> multiply(
    const Matrix<S, OA, AllocA>& A,
    const Matrix<S, OB, AllocB>& B) {
    Matrix<S, OC, AllocC> C(A.rows(), B.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t k = 0; k < A.cols(); ++k) {
            for (size_t j = 0; j < B.cols(); ++j) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    return C;
}

template<typename S, StorageOrder OA, StorageOrder OB, template<typename> class AllocA, template<typename> class AllocB>
Matrix<S, OA, AllocA> multiply(
    const Matrix<S, OA, AllocA>& A,
    const Matrix<S, OB, AllocB>& B) {
    return multiply<S, OA, OB, OA, AllocA, AllocB, AllocA>(A, B);
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, StorageOrder::RowMajor, Alloc> transpose_copy(
    const Matrix<S, OA, Alloc>& A) {
    Matrix<S, StorageOrder::RowMajor, Alloc> T(A.cols(), A.rows());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            T(j, i) = A(i, j);
        }
    }
    return T;
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, StorageOrder::ColMajor, Alloc> to_col_major(const Matrix<S, OA, Alloc>& A) {
    Matrix<S, StorageOrder::ColMajor, Alloc> B(A.rows(), A.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            B(i, j) = A(i, j);
        }
    }
    return B;
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
S frobenius_norm(const Matrix<S, OA, Alloc>& A) {
    S sum = S(0);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            sum += A(i, j) * A(i, j);
        }
    }
    return std::sqrt(sum);
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
bool is_symmetric(const Matrix<S, OA, Alloc>& A, S tol = S(1e-10)) {
    if (A.rows() != A.cols()) {
        return false;
    }
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = i + 1; j < A.cols(); ++j) {
            if (std::abs(A(i, j) - A(j, i)) > tol) {
                return false;
            }
        }
    }
    return true;
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
S dot_col(const Matrix<S, OA, Alloc>& A, size_t j, size_t k) {
    S sum = S(0);
    for (size_t i = 0; i < A.rows(); ++i) {
        sum += A(i, j) * A(i, k);
    }
    return sum;
}

} // namespace ms::linalg_detail
