#include "ms/linalg/linalg.hpp"
#include "ms/cpu/lapack.hpp"
#include "detail.hpp"
#include <cmath>
#include <limits>

namespace ms {

namespace {

using namespace linalg_detail;

} // namespace

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<S> rank(const Matrix<S, OA, Alloc>& A, S tol) {
    auto svd_result = svd(A);
    if (!svd_result) {
        return std::unexpected(svd_result.error());
    }

    if (tol == S(0)) {
        tol = S(1e-10) * std::max<S>(A.rows(), A.cols()) * svd_result->S(0, 0);
    }

    size_t r = 0;
    for (size_t i = 0; i < svd_result->S.rows(); ++i) {
        if (svd_result->S(i, 0) > tol) {
            ++r;
        }
    }
    return static_cast<S>(r);
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<S> cond(const Matrix<S, OA, Alloc>& A, int p) {
    if (p != 2) {
        return std::unexpected(DomainError{"cond", "only p=2 supported"});
    }

    auto svd_result = svd(A);
    if (!svd_result) {
        return std::unexpected(svd_result.error());
    }

    double smax = 0.0;
    double smin = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < svd_result->S.rows(); ++i) {
        smax = std::max(smax, svd_result->S(i, 0));
        if (svd_result->S(i, 0) > 1e-14) {
            smin = std::min(smin, svd_result->S(i, 0));
        }
    }
    if (smin == std::numeric_limits<double>::infinity()) {
        return std::unexpected(SingularMatrix{});
    }
    return smax / smin;
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> lsq(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b) {
    if (A.rows() != b.rows()) {
        return std::unexpected(DimensionMismatch{A.rows(), b.rows()});
    }

    if constexpr (std::is_same_v<S, double> && OA == StorageOrder::ColMajor) {
        if (A.rows() >= A.cols() && A.cols() > 0) {
            const int m = static_cast<int>(A.rows());
            const int n = static_cast<int>(A.cols());
            const int nrhs = static_cast<int>(b.cols());
            Matrix<double> factor = copy(A);
            Matrix<double> rhs = copy(b);
            if (cpu::lapack::dgels(m, n, nrhs, factor.data(), m, rhs.data(), m) == 0) {
                Matrix<S, OA, Alloc> x(static_cast<std::size_t>(n), static_cast<std::size_t>(nrhs));
                for (int j = 0; j < nrhs; ++j) {
                    for (int i = 0; i < n; ++i) {
                        x(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                            rhs(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
                    }
                }
                return x;
            }
        }
    }

    auto svd_result = svd(A);
    if (!svd_result) {
        return std::unexpected(svd_result.error());
    }

    const auto& U = svd_result->U;
    const auto& sigma = svd_result->S;
    const auto& V = svd_result->V;

    auto Ut = transpose_copy(U);
    auto Utb = multiply(Ut, b);
    Matrix<double> Sinv(sigma.rows(), 1);
    for (size_t i = 0; i < sigma.rows(); ++i) {
        Sinv(i, 0) = sigma(i, 0) > 1e-14 ? 1.0 / sigma(i, 0) : 0.0;
    }

    Matrix<double> weighted(Utb.rows(), Utb.cols());
    for (size_t i = 0; i < Utb.rows(); ++i) {
        for (size_t j = 0; j < Utb.cols(); ++j) {
            weighted(i, j) = Sinv(i, 0) * Utb(i, j);
        }
    }

    return multiply(V, weighted);
}

template auto rank<double>(const Matrix<double>&, double) -> Result<double>;
template auto cond<double>(const Matrix<double>&, int) -> Result<double>;
template auto lsq<double>(const Matrix<double>&, const Matrix<double>&) -> Result<Matrix<double>>;

} // namespace ms
