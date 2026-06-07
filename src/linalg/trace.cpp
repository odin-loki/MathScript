#include "ms/core/operations.hpp"

namespace ms {

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<S> trace(const Matrix<S, OA, Alloc>& A) {
    S result = S(0);
    const size_t n = std::min(A.rows(), A.cols());
    for (size_t i = 0; i < n; ++i) {
        result += A(i, i);
    }
    return result;
}

template auto trace<double>(const Matrix<double>&) -> Result<double>;
template auto trace<float>(const Matrix<float>&) -> Result<float>;

} // namespace ms
