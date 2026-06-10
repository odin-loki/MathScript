#include "ms/core/operations.hpp"

namespace ms {

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<S> trace(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }
    S result = S(0);
    for (size_t i = 0; i < A.rows(); ++i) {
        result += A(i, i);
    }
    return result;
}

template auto trace<double>(const Matrix<double>&) -> Result<double>;
template auto trace<float>(const Matrix<float>&) -> Result<float>;

} // namespace ms
