#include "ms/core/operations.hpp"
#include "ms/cpu/lapack.hpp"
#include "ms/simd/simd.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace ms {

namespace {

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<std::tuple<Matrix<S, OA, Alloc>, Matrix<S, OA, Alloc>>>
qr_modified_gram_schmidt(const Matrix<S, OA, Alloc>& A) {
    const size_t m = A.rows();
    const size_t n = A.cols();
    Matrix<S, OA, Alloc> Q(m, n, S(0));
    Matrix<S, OA, Alloc> R(n, n, S(0));
    std::vector<double> v(m);

    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            v[i] = A(i, j);
        }

        for (size_t i = 0; i < j; ++i) {
            double* q_col = Q.data() + i * m;
            const double rij = ms::simd::dot({q_col, m}, {v.data(), m});
            R(i, j) = static_cast<S>(rij);
            ms::simd::axpy(-rij, {q_col, m}, {v.data(), m});
        }

        const double norm_v = std::sqrt(ms::simd::dot({v.data(), m}, {v.data(), m}));
        R(j, j) = static_cast<S>(norm_v);
        if (norm_v < std::numeric_limits<double>::epsilon()) {
            continue;
        }

        double* q_col = Q.data() + j * m;
        for (size_t k = 0; k < m; ++k) {
            q_col[k] = v[k] / norm_v;
        }
    }

    return std::make_tuple(Q, R);
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<std::tuple<Matrix<S, OA, Alloc>, Matrix<S, OA, Alloc>>>
qr_householder(const Matrix<S, OA, Alloc>& A) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    Matrix<S, OA, Alloc> work = A;
    std::vector<double> tau(static_cast<std::size_t>((std::min)(m, n)));

    if (cpu::lapack::dgeqrf(m, n, work.data(), m, tau.data()) != 0) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }

    Matrix<S, OA, Alloc> R(static_cast<std::size_t>(n), static_cast<std::size_t>(n), S(0));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i <= j && i < m; ++i) {
            R(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                work(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        }
    }

    Matrix<S, OA, Alloc> Q(static_cast<std::size_t>(m), static_cast<std::size_t>(n), S(0));
    cpu::lapack::dorgqr(m, n, n, work.data(), m, tau.data(), Q.data(), m);
    return std::make_tuple(Q, R);
}

} // namespace

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<std::tuple<Matrix<S, OA, Alloc>, Matrix<S, OA, Alloc>>>
qr(const Matrix<S, OA, Alloc>& A) {
    const size_t m = A.rows();
    const size_t n = A.cols();
    if (m == 0 || n == 0) {
        return std::unexpected(DimensionMismatch{m, n});
    }

    if constexpr (std::is_same_v<S, double> && OA == StorageOrder::ColMajor) {
        if (m >= n) {
            return qr_householder(A);
        }
        return qr_modified_gram_schmidt(A);
    }

    Matrix<S, OA, Alloc> Q(m, n, S(0));
    Matrix<S, OA, Alloc> R(n, n, S(0));

    for (size_t j = 0; j < n; ++j) {
        Matrix<S, OA, Alloc> v(m, 1);
        for (size_t i = 0; i < m; ++i) {
            v(i, 0) = A(i, j);
        }

        for (size_t i = 0; i < j; ++i) {
            S dot = S(0);
            for (size_t k = 0; k < m; ++k) {
                dot += Q(k, i) * v(k, 0);
            }
            R(i, j) = dot;
            for (size_t k = 0; k < m; ++k) {
                v(k, 0) -= R(i, j) * Q(k, i);
            }
        }

        S norm_v = S(0);
        for (size_t k = 0; k < m; ++k) {
            norm_v += v(k, 0) * v(k, 0);
        }
        norm_v = std::sqrt(norm_v);
        R(j, j) = norm_v;

        if (norm_v < std::numeric_limits<S>::epsilon()) {
            for (size_t k = 0; k < m; ++k) {
                Q(k, j) = S(0);
            }
            continue;
        }

        for (size_t k = 0; k < m; ++k) {
            Q(k, j) = v(k, 0) / norm_v;
        }
    }

    return std::make_tuple(Q, R);
}

template auto qr<double>(const Matrix<double>&)
    -> Result<std::tuple<Matrix<double>, Matrix<double>>>;
template auto qr<float>(const Matrix<float>&)
    -> Result<std::tuple<Matrix<float>, Matrix<float>>>;

} // namespace ms
