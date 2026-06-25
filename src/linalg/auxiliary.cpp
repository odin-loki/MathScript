#include "ms/linalg/linalg.hpp"
#include "ms/cpu/lapack.hpp"
#include "detail.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

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

// --- pinv ---
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> pinv(const Matrix<S, OA, Alloc>& A, S tol) {
    auto svd_res = svd(A);
    if (!svd_res) return std::unexpected(svd_res.error());
    const auto& U = svd_res->U;    // m x k
    const auto& sigma = svd_res->S; // k x 1
    const auto& V = svd_res->V;    // n x k (in thin SVD, n x n)
    if (tol == S(0)) {
        S smax = sigma(0, 0);
        tol = S(1e-10) * static_cast<S>(std::max(A.rows(), A.cols())) * smax;
    }
    // pinv = V * diag(1/sigma) * U^T
    // Compute manually: pinv(i,j) = sum_k V(i,k) * (1/sigma_k) * U(j,k)
    const size_t m = A.rows(), n = A.cols();
    const size_t k = sigma.rows();
    Matrix<S, OA, Alloc> P(n, m, S(0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            S val = S(0);
            for (size_t r = 0; r < k && r < V.cols() && r < U.cols(); ++r) {
                if (sigma(r, 0) > tol)
                    val += V(i, r) * (S(1) / sigma(r, 0)) * U(j, r);
            }
            P(i, j) = val;
        }
    }
    return P;
}

// --- null ---
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> null(const Matrix<S, OA, Alloc>& A, S tol) {
    const size_t m = A.rows();
    const size_t n = A.cols();
    if (n == 0) {
        return Matrix<S, OA, Alloc>(0, 0, S(0));
    }

    // Null space of A: eigenvectors of A^T A with near-zero eigenvalue (sigma^2).
    // Thin SVD V is n x min(m,n) and cannot span the full null space when m < n.
    auto gram = multiply(transpose_copy(A), A);
    auto eig_res = eig_sym(gram);
    if (!eig_res) {
        return std::unexpected(eig_res.error());
    }

    S emax = S(0);
    for (size_t i = 0; i < n; ++i) {
        emax = (std::max)(emax, static_cast<S>(eig_res->values(i, 0)));
    }
    if (tol == S(0)) {
        const S sigma_max = std::sqrt((std::max)(emax, S(0)));
        tol = S(1e-10) * static_cast<S>(std::max(m, n)) * sigma_max;
    }
    const S tol_sq = tol * tol;

    std::vector<size_t> null_cols;
    for (size_t i = 0; i < n; ++i) {
        if (eig_res->values(i, 0) <= tol_sq * emax) {
            null_cols.push_back(i);
        }
    }
    if (null_cols.empty()) {
        return Matrix<S, OA, Alloc>(n, 0, S(0));
    }

    Matrix<S, OA, Alloc> N(n, null_cols.size(), S(0));
    for (size_t j = 0; j < null_cols.size(); ++j) {
        const size_t col = null_cols[j];
        for (size_t i = 0; i < n; ++i) {
            N(i, j) = eig_res->vectors(i, col);
        }
    }
    return N;
}

// --- orth ---
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> orth(const Matrix<S, OA, Alloc>& A, S tol) {
    if (A.rows() == 0 || A.cols() == 0) {
        return Matrix<S, OA, Alloc>(A.rows(), 0, S(0));
    }
    auto qr_res = qr(A);
    if (!qr_res) return std::unexpected(qr_res.error());
    const auto& Q = std::get<0>(*qr_res);
    const auto& R = std::get<1>(*qr_res);
    const size_t k = (std::min)(R.rows(), R.cols());
    if (tol == S(0)) {
        S rmax = S(0);
        for (size_t i = 0; i < k; ++i) {
            rmax = (std::max)(rmax, static_cast<S>(std::abs(R(i, i))));
        }
        tol = S(1e-10) * static_cast<S>(std::max(A.rows(), A.cols())) * rmax;
    }
    size_t r = 0;
    for (size_t i = 0; i < k; ++i) {
        if (std::abs(R(i, i)) > tol) ++r;
    }
    const size_t m = Q.rows();
    Matrix<S, OA, Alloc> Qout(m, r, S(0));
    for (size_t j = 0; j < r; ++j) {
        for (size_t i = 0; i < m; ++i) {
            Qout(i, j) = Q(i, j);
        }
    }
    return Qout;
}

template auto pinv<double>(const Matrix<double>&, double) -> Result<Matrix<double>>;
template auto null<double>(const Matrix<double>&, double) -> Result<Matrix<double>>;
template auto orth<double>(const Matrix<double>&, double) -> Result<Matrix<double>>;

} // namespace ms
