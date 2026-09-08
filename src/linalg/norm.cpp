#include "ms/core/operations.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace ms {

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<S> norm(const Matrix<S, OA, Alloc>& A, int p) {
    // p == -1 spells the induced infinity norm (largest absolute row sum).
    if (p == -1) {
        S max_row = S(0);
        for (size_t i = 0; i < A.rows(); ++i) {
            S row_sum = S(0);
            for (size_t j = 0; j < A.cols(); ++j) {
                row_sum += std::abs(A(i, j));
            }
            max_row = std::max(max_row, row_sum);
        }
        return max_row;
    }

    // p == 0 is the l0 "norm": how many entries are nonzero.
    if (p == 0) {
        S count = S(0);
        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t j = 0; j < A.cols(); ++j) {
                if (A(i, j) != S(0)) {
                    count += S(1);
                }
            }
        }
        return count;
    }

    if (p == 1) {
        S sum = S(0);
        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t j = 0; j < A.cols(); ++j) {
                sum += std::abs(A(i, j));
            }
        }
        return sum;
    }

    if (p == 2) {
        // Frobenius / Euclidean norm, accumulated on the largest entry so a
        // matrix of huge or tiny entries does not overflow the sum of squares.
        S scale = S(0);
        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t j = 0; j < A.cols(); ++j) {
                scale = std::max(scale, std::abs(A(i, j)));
            }
        }
        if (scale == S(0)) {
            return S(0);
        }
        S sum = S(0);
        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t j = 0; j < A.cols(); ++j) {
                const S t = A(i, j) / scale;
                sum += t * t;
            }
        }
        return scale * std::sqrt(sum);
    }

    if (p > 2) {
        S scale = S(0);
        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t j = 0; j < A.cols(); ++j) {
                scale = std::max(scale, std::abs(A(i, j)));
            }
        }
        if (scale == S(0)) {
            return S(0);
        }
        S sum = S(0);
        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t j = 0; j < A.cols(); ++j) {
                sum += std::pow(std::abs(A(i, j)) / scale, static_cast<S>(p));
            }
        }
        return scale * std::pow(sum, S(1) / static_cast<S>(p));
    }

    // Everything else (p <= -2) has no definition here. Report it instead of
    // handing back a plausible-looking zero, matching cond()'s convention.
    return std::unexpected(DomainError{
        "norm", "unsupported p (supported: 0, 1, 2, any p > 2, and -1 for infinity)"});
}

template auto norm<double>(const Matrix<double>&, int) -> Result<double>;
template auto norm<float>(const Matrix<float>&, int) -> Result<float>;

} // namespace ms
