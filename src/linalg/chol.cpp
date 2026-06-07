#include "ms/core/operations.hpp"
#include "ms/cpu/lapack.hpp"
#include <cmath>

namespace ms {

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> chol(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }

    if constexpr (std::is_same_v<S, double> && OA == StorageOrder::ColMajor) {
        const int n = static_cast<int>(A.rows());
        Matrix<S, OA, Alloc> factor = A;
        const int info = cpu::lapack::dpotrf('L', n, factor.data(), n);
        if (info != 0) {
            return std::unexpected(SingularMatrix{});
        }

        Matrix<S, OA, Alloc> L(static_cast<std::size_t>(n), static_cast<std::size_t>(n), S(0));
        for (int j = 0; j < n; ++j) {
            for (int i = j; i < n; ++i) {
                L(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                    factor(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
            }
        }
        return L;
    }

    const size_t n = A.rows();
    Matrix<S, OA, Alloc> L(n, n, S(0));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            S sum = S(0);
            for (size_t k = 0; k < j; ++k) {
                sum += L(i, k) * L(j, k);
            }
            if (i == j) {
                S val = A(i, i) - sum;
                if (val <= S(0)) {
                    return std::unexpected(SingularMatrix{});
                }
                L(i, j) = std::sqrt(val);
            } else {
                L(i, j) = (A(i, j) - sum) / L(j, j);
            }
        }
    }

    return L;
}

template auto chol<double>(const Matrix<double>&) -> Result<Matrix<double>>;
template auto chol<float>(const Matrix<float>&) -> Result<Matrix<float>>;

} // namespace ms
