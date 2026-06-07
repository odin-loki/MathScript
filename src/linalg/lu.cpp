#include "ms/core/operations.hpp"
#include "ms/cpu/lapack.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace ms {

namespace {

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<std::tuple<Matrix<S, OA, Alloc>, Matrix<S, OA, Alloc>, Matrix<S, OA, Alloc>>>
extract_lu_factors(const Matrix<S, OA, Alloc>& factor, const std::vector<int>& ipiv) {
    const size_t n = factor.rows();
    Matrix<S, OA, Alloc> P(n, n, S(0));
    for (size_t i = 0; i < n; ++i) {
        P(i, i) = S(1);
    }
    for (size_t k = 0; k < n; ++k) {
        if (ipiv[k] != static_cast<int>(k)) {
            for (size_t j = 0; j < n; ++j) {
                std::swap(P(k, j), P(static_cast<std::size_t>(ipiv[k]), j));
            }
        }
    }

    Matrix<S, OA, Alloc> L(n, n, S(0));
    Matrix<S, OA, Alloc> U(n, n, S(0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i > j) {
                L(i, j) = factor(i, j);
            } else if (i == j) {
                L(i, j) = S(1);
                U(i, j) = factor(i, j);
            } else {
                U(i, j) = factor(i, j);
            }
        }
    }

    return std::make_tuple(L, U, P);
}

} // namespace

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<std::tuple<Matrix<S, OA, Alloc>, Matrix<S, OA, Alloc>, Matrix<S, OA, Alloc>>>
lu(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }

    if constexpr (std::is_same_v<S, double> && OA == StorageOrder::ColMajor) {
        const int n = static_cast<int>(A.rows());
        Matrix<S, OA, Alloc> factor = A;
        std::vector<int> ipiv(static_cast<std::size_t>(n));
        std::iota(ipiv.begin(), ipiv.end(), 0);
        const int info = cpu::lapack::dgetrf(n, n, factor.data(), n, ipiv.data());
        if (info != 0) {
            return std::unexpected(SingularMatrix{});
        }
        return extract_lu_factors(factor, ipiv);
    }

    const size_t n = A.rows();
    Matrix<S, OA, Alloc> LU = A;
    Matrix<S, OA, Alloc> P(n, n, S(0));
    std::vector<size_t> perm(n);

    for (size_t i = 0; i < n; ++i) {
        perm[i] = i;
        for (size_t j = 0; j < n; ++j) {
            P(i, j) = (i == j) ? S(1) : S(0);
        }
    }

    for (size_t k = 0; k < n; ++k) {
        size_t pivot = k;
        S max_val = std::abs(LU(k, k));
        for (size_t i = k + 1; i < n; ++i) {
            const S val = std::abs(LU(i, k));
            if (val > max_val) {
                max_val = val;
                pivot = i;
            }
        }

        if (max_val < std::numeric_limits<S>::epsilon()) {
            return std::unexpected(SingularMatrix{});
        }

        if (pivot != k) {
            for (size_t j = 0; j < n; ++j) {
                std::swap(LU(k, j), LU(pivot, j));
                std::swap(P(k, j), P(pivot, j));
            }
        }

        for (size_t i = k + 1; i < n; ++i) {
            LU(i, k) /= LU(k, k);
            for (size_t j = k + 1; j < n; ++j) {
                LU(i, j) -= LU(i, k) * LU(k, j);
            }
        }
    }

    Matrix<S, OA, Alloc> L(n, n, S(0));
    Matrix<S, OA, Alloc> U(n, n, S(0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i > j) {
                L(i, j) = LU(i, j);
            } else if (i == j) {
                L(i, j) = S(1);
                U(i, j) = LU(i, j);
            } else {
                U(i, j) = LU(i, j);
            }
        }
    }

    return std::make_tuple(L, U, P);
}

template auto lu<double>(const Matrix<double>&)
    -> Result<std::tuple<Matrix<double>, Matrix<double>, Matrix<double>>>;
template auto lu<float>(const Matrix<float>&)
    -> Result<std::tuple<Matrix<float>, Matrix<float>, Matrix<float>>>;

} // namespace ms
