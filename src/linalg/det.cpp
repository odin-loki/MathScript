#include "ms/core/operations.hpp"
#include <cmath>

namespace ms {

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<S> det(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }

    auto lu_result = lu(A);
    if (!lu_result) {
        return std::unexpected(lu_result.error());
    }

    const auto& [L, U, P] = *lu_result;
    const size_t n = A.rows();
    S det_val = S(1);
    for (size_t i = 0; i < n; ++i) {
        det_val *= U(i, i);
    }

    S sign = S(1);
    for (size_t k = 0; k < n; ++k) {
        if (P(k, k) != S(1)) {
            sign = -sign;
        }
    }

    return det_val * sign;
}

template auto det<double>(const Matrix<double>&) -> Result<double>;
template auto det<float>(const Matrix<float>&) -> Result<float>;

} // namespace ms
