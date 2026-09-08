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
        tol = S(1e-10) * std::max(static_cast<S>(A.rows()), static_cast<S>(A.cols())) * svd_result->S(0, 0);
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
int matrix_rank(const Matrix<S, OA, Alloc>& A, double tol) {
    auto svd_result = svd(A);
    if (!svd_result) {
        return 0;
    }

    int r = 0;
    for (size_t i = 0; i < svd_result->S.rows(); ++i) {
        if (svd_result->S(i, 0) > tol) {
            ++r;
        }
    }
    return r;
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
template auto matrix_rank<double>(const Matrix<double>&, double) -> int;
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

    // Tall/square case: the SVD's V is n x n and its singular values come
    // straight from LAPACK rather than from a squared Gram matrix, so the rank
    // (and therefore the nullity) is read off sigma directly.
    //
    // The trailing columns of V cannot simply be handed back as the basis:
    // svd() leaves the right singular vector of an exactly-zero singular value
    // as a zero column (the LAPACK path never fills it, and the Gram fallback
    // skips it explicitly at svd.cpp's `sigma < 1e-14` guard).  A zero column
    // satisfies A*v == 0 vacuously, so it would pass every residual check while
    // carrying no direction at all.  The basis is therefore built by
    // orthonormal completion: span the ROW space with the reliable
    // (sigma > cutoff) right singular vectors, then fill the orthogonal
    // complement -- preferring the SVD's own trailing vectors whenever they are
    // usable, and falling back to coordinate directions when they are not.
    if (m >= n) {
        auto svd_res = svd(A);
        if (svd_res) {
            const auto& sigma = svd_res->S;
            const auto& V = svd_res->V;
            if (V.rows() == n && V.cols() == n && sigma.rows() == n) {
                S cutoff = tol;
                if (cutoff == S(0)) {
                    cutoff = S(1e-10) * static_cast<S>(std::max(m, n))
                           * static_cast<S>(sigma(0, 0));
                }
                size_t rank_r = 0;
                for (size_t i = 0; i < n; ++i) {
                    if (static_cast<S>(sigma(i, 0)) > cutoff) {
                        ++rank_r;
                    }
                }
                const size_t nullity = n - rank_r;
                if (nullity == 0) {
                    return Matrix<S, OA, Alloc>(n, 0, S(0));
                }

                // basis holds an orthonormal set: first the row-space
                // directions, then the accepted null-space directions.
                std::vector<std::vector<S>> basis;
                basis.reserve(n);
                // Orthogonalise v against `basis` twice (classical
                // Gram-Schmidt twice is as accurate as modified GS here) and
                // return the residual norm.
                auto reduce = [&basis, n](std::vector<S>& v) -> S {
                    for (int pass = 0; pass < 2; ++pass) {
                        for (const auto& q : basis) {
                            S d = S(0);
                            for (size_t i = 0; i < n; ++i) {
                                d += q[i] * v[i];
                            }
                            for (size_t i = 0; i < n; ++i) {
                                v[i] -= d * q[i];
                            }
                        }
                    }
                    S s = S(0);
                    for (size_t i = 0; i < n; ++i) {
                        s += v[i] * v[i];
                    }
                    return std::sqrt(s);
                };
                auto push_normalised = [&basis, n](std::vector<S>& v, S nrm) {
                    for (size_t i = 0; i < n; ++i) {
                        v[i] /= nrm;
                    }
                    basis.push_back(v);
                };

                // 1. Row space: the sigma > cutoff right singular vectors.
                for (size_t i = 0; i < n && basis.size() < rank_r; ++i) {
                    if (!(static_cast<S>(sigma(i, 0)) > cutoff)) {
                        continue;
                    }
                    std::vector<S> v(n);
                    for (size_t k = 0; k < n; ++k) {
                        v[k] = static_cast<S>(V(k, i));
                    }
                    const S nrm = reduce(v);
                    if (nrm > S(0.5)) {
                        push_normalised(v, nrm);
                    }
                }
                // Any singular vector that came back degenerate is replaced by
                // the row of A that is furthest from the span built so far;
                // rows of A live in the row space, so the span stays correct.
                while (basis.size() < rank_r) {
                    std::vector<S> best;
                    S best_nrm = S(0);
                    for (size_t i = 0; i < m; ++i) {
                        std::vector<S> row(n);
                        for (size_t k = 0; k < n; ++k) {
                            row[k] = A(i, k);
                        }
                        const S nrm = reduce(row);
                        if (nrm > best_nrm) {
                            best_nrm = nrm;
                            best = row;
                        }
                    }
                    if (best.empty() || !(best_nrm > S(0))) {
                        break;
                    }
                    push_normalised(best, best_nrm);
                }

                // 2. Null space: the remaining singular vectors when usable.
                const size_t row_space = basis.size();
                for (size_t i = 0; i < n && basis.size() < row_space + nullity; ++i) {
                    if (static_cast<S>(sigma(i, 0)) > cutoff) {
                        continue;
                    }
                    std::vector<S> v(n);
                    for (size_t k = 0; k < n; ++k) {
                        v[k] = static_cast<S>(V(k, i));
                    }
                    const S nrm = reduce(v);
                    if (nrm > S(0.5)) {
                        push_normalised(v, nrm);
                    }
                }
                // 3. Complete with the coordinate direction that is furthest
                //    from the current span (its residual is at least
                //    sqrt((n - |basis|)/n), so this always terminates). A
                //    candidate that is already essentially orthogonal to the
                //    span is taken immediately -- it is just as good a basis
                //    vector as the maximiser, and stopping early keeps the
                //    common case (few, well-separated directions) at O(n^3).
                while (basis.size() < row_space + nullity) {
                    std::vector<S> best;
                    S best_nrm = S(0);
                    for (size_t j = 0; j < n; ++j) {
                        std::vector<S> e(n, S(0));
                        e[j] = S(1);
                        const S nrm = reduce(e);
                        if (nrm > best_nrm) {
                            best_nrm = nrm;
                            best = e;
                        }
                        if (best_nrm > S(0.9)) {
                            break;
                        }
                    }
                    if (best.empty() || !(best_nrm > S(0))) {
                        break;
                    }
                    push_normalised(best, best_nrm);
                }

                const size_t got = basis.size() - row_space;
                Matrix<S, OA, Alloc> N(n, got, S(0));
                for (size_t j = 0; j < got; ++j) {
                    for (size_t i = 0; i < n; ++i) {
                        N(i, j) = basis[row_space + j][i];
                    }
                }
                return N;
            }
        }
    }

    // Wide case (m < n): the thin SVD's V is only n x m and cannot span an
    // (n - m)-dimensional null space, so fall back to the eigenvectors of
    // A^T A with a near-zero eigenvalue (sigma^2).
    auto gram = multiply(transpose_copy(A), A);
    auto eig_res = eig_sym(gram);
    if (!eig_res) {
        return std::unexpected(eig_res.error());
    }

    S emax = S(0);
    for (size_t i = 0; i < n; ++i) {
        emax = (std::max)(emax, static_cast<S>(eig_res->values(i, 0)));
    }
    S cutoff = tol;
    if (cutoff == S(0)) {
        // Forming A^T A squares the condition number, so a singular value that
        // is mathematically zero only comes back at the sqrt(eps) level; the
        // default cutoff has to sit above that, unlike the SVD path above.
        const S sigma_max = std::sqrt((std::max)(emax, S(0)));
        cutoff = std::sqrt(std::numeric_limits<S>::epsilon())
               * static_cast<S>(std::max(m, n)) * sigma_max;
    }
    // Eigenvalues of A^T A are sigma^2, and `cutoff` is in singular-value
    // units, so the comparison is against cutoff^2 -- with no second factor
    // of sigma_max, which would make the test scale-dependent.
    const S cutoff_sq = cutoff * cutoff;

    std::vector<size_t> null_cols;
    for (size_t i = 0; i < n; ++i) {
        if (static_cast<S>(eig_res->values(i, 0)) <= cutoff_sq) {
            null_cols.push_back(i);
        }
    }
    if (null_cols.empty()) {
        return Matrix<S, OA, Alloc>(n, 0, S(0));
    }

    Matrix<S, OA, Alloc> N(n, null_cols.size(), S(0));
    for (size_t j = 0; j < null_cols.size(); ++j) {
        const size_t col = null_cols[j];
        S nrm = S(0);
        for (size_t i = 0; i < n; ++i) {
            nrm += static_cast<S>(eig_res->vectors(i, col))
                 * static_cast<S>(eig_res->vectors(i, col));
        }
        nrm = std::sqrt(nrm);
        if (nrm == S(0)) {
            nrm = S(1);
        }
        for (size_t i = 0; i < n; ++i) {
            N(i, j) = static_cast<S>(eig_res->vectors(i, col)) / nrm;
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
    // An unpivoted QR is not rank revealing: taking the leading columns of its
    // Q returns directions outside range(A) whenever the deficiency is not in
    // the trailing columns. The SVD is, so use it (this is also what MATLAB's
    // orth does, and it matches rank/pinv/cond in this file).
    auto svd_res = svd(A);
    if (!svd_res) {
        return std::unexpected(svd_res.error());
    }
    const auto& U = svd_res->U;
    const auto& sigma = svd_res->S;
    if (tol == S(0)) {
        tol = S(1e-10) * static_cast<S>(std::max(A.rows(), A.cols()))
            * static_cast<S>(sigma(0, 0));
    }
    size_t r = 0;
    for (size_t i = 0; i < sigma.rows() && i < U.cols(); ++i) {
        if (static_cast<S>(sigma(i, 0)) > tol) {
            ++r;
        }
    }
    const size_t m = U.rows();
    Matrix<S, OA, Alloc> Qout(m, r, S(0));
    for (size_t j = 0; j < r; ++j) {
        for (size_t i = 0; i < m; ++i) {
            Qout(i, j) = static_cast<S>(U(i, j));
        }
    }
    return Qout;
}

template auto pinv<double>(const Matrix<double>&, double) -> Result<Matrix<double>>;
template auto null<double>(const Matrix<double>&, double) -> Result<Matrix<double>>;
template auto orth<double>(const Matrix<double>&, double) -> Result<Matrix<double>>;

} // namespace ms
