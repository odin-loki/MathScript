#include "ms/core/operations.hpp"

namespace ms {

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> expm(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }

    const size_t n = A.rows();
    Matrix<S, OA, Alloc> result = eye<S, Alloc>(n);
    Matrix<S, OA, Alloc> term = eye<S, Alloc>(n);
    const int terms = 12;

    for (int k = 1; k <= terms; ++k) {
        Matrix<S, OA, Alloc> next(n, n, S(0));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                for (size_t m = 0; m < n; ++m) {
                    next(i, j) += term(i, m) * A(m, j);
                }
            }
        }
        const S scale = S(1) / S(k);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                term(i, j) = next(i, j) * scale;
                result(i, j) += term(i, j);
            }
        }
    }

    return result;
}

template auto expm<double>(const Matrix<double>&) -> Result<Matrix<double>>;
template auto expm<float>(const Matrix<float>&) -> Result<Matrix<float>>;

} // namespace ms
