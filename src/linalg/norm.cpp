#include "ms/core/operations.hpp"
#include <cmath>
#include <limits>

namespace ms {

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<S> norm(const Matrix<S, OA, Alloc>& A, int p) {
    if (p == 2) {
        S sum = S(0);
        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t j = 0; j < A.cols(); ++j) {
                sum += A(i, j) * A(i, j);
            }
        }
        return std::sqrt(sum);
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

    return S(0);
}

template auto norm<double>(const Matrix<double>&, int) -> Result<double>;
template auto norm<float>(const Matrix<float>&, int) -> Result<float>;

} // namespace ms
