// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "ms/linalg/linalg.hpp"
#include "detail.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace ms {

namespace {

using namespace linalg_detail;

// ---------------------------------------------------------------------------
// Householder machinery
//
// A reflector is represented by (v, beta) with P = I - beta*v*v^T acting on a
// contiguous index range [off, off + v.size()). P is symmetric and orthogonal,
// so P^-1 == P^T == P.
// ---------------------------------------------------------------------------

// Build the reflector that maps A(start..n-1, col) onto alpha*e_1.
// Returns false when A(start+1..n-1, col) is already exactly zero, i.e. there
// is nothing to annihilate and the identity reflector would do.
bool house_col(const Matrix<double>& A, size_t start, size_t col,
               std::vector<double>& v, double& beta, double& alpha) {
    const size_t n = A.rows();
    if (start + 1 >= n) {
        return false;
    }
    double tail_sq = 0.0;
    for (size_t i = start + 1; i < n; ++i) {
        tail_sq += A(i, col) * A(i, col);
    }
    if (tail_sq == 0.0) {
        return false;
    }
    const double head = A(start, col);
    const double scale = std::sqrt(head * head + tail_sq);
    const double sign = (head >= 0.0) ? 1.0 : -1.0;
    alpha = -sign * scale;

    v.assign(n - start, 0.0);
    v[0] = head + sign * scale;
    for (size_t i = start + 1; i < n; ++i) {
        v[i - start] = A(i, col);
    }
    double vtv = 0.0;
    for (double x : v) {
        vtv += x * x;
    }
    if (vtv == 0.0) {
        return false;
    }
    beta = 2.0 / vtv;
    return true;
}

// Row analogue: build the reflector that maps A(row, start..m-1) onto
// alpha*e_1^T (used for the right-hand side of the bidiagonalisation).
bool house_row(const Matrix<double>& A, size_t row, size_t start,
               std::vector<double>& v, double& beta, double& alpha) {
    const size_t m = A.cols();
    if (start + 1 >= m) {
        return false;
    }
    double tail_sq = 0.0;
    for (size_t j = start + 1; j < m; ++j) {
        tail_sq += A(row, j) * A(row, j);
    }
    if (tail_sq == 0.0) {
        return false;
    }
    const double head = A(row, start);
    const double scale = std::sqrt(head * head + tail_sq);
    const double sign = (head >= 0.0) ? 1.0 : -1.0;
    alpha = -sign * scale;

    v.assign(m - start, 0.0);
    v[0] = head + sign * scale;
    for (size_t j = start + 1; j < m; ++j) {
        v[j - start] = A(row, j);
    }
    double vtv = 0.0;
    for (double x : v) {
        vtv += x * x;
    }
    if (vtv == 0.0) {
        return false;
    }
    beta = 2.0 / vtv;
    return true;
}

// A <- P*A, with P acting on rows [off, off+v.size()); columns [j0, cols).
void apply_left(Matrix<double>& A, size_t off, const std::vector<double>& v,
                double beta, size_t j0) {
    const size_t ncols = A.cols();
    const size_t len = v.size();
    for (size_t j = j0; j < ncols; ++j) {
        double dot = 0.0;
        for (size_t i = 0; i < len; ++i) {
            dot += v[i] * A(off + i, j);
        }
        dot *= beta;
        for (size_t i = 0; i < len; ++i) {
            A(off + i, j) -= dot * v[i];
        }
    }
}

// A <- A*P, with P acting on columns [off, off+v.size()); rows [i0, rows).
void apply_right(Matrix<double>& A, size_t off, const std::vector<double>& v,
                 double beta, size_t i0) {
    const size_t nrows = A.rows();
    const size_t len = v.size();
    for (size_t i = i0; i < nrows; ++i) {
        double dot = 0.0;
        for (size_t j = 0; j < len; ++j) {
            dot += A(i, off + j) * v[j];
        }
        dot *= beta;
        for (size_t j = 0; j < len; ++j) {
            A(i, off + j) -= dot * v[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Hessenberg reduction (a genuine orthogonal similarity)
//
// On return H = Q^T * A_in * Q (equivalently A_in = Q*H*Q^T) with H upper
// Hessenberg and Q orthogonal. Each reflector is applied from BOTH sides,
// which is what makes the trace, determinant and spectrum invariant.
// ---------------------------------------------------------------------------
void hess_reduce(Matrix<double>& H, Matrix<double>& Q) {
    const size_t n = H.rows();
    std::vector<double> v;
    double beta = 0.0;
    double alpha = 0.0;
    for (size_t k = 0; k + 2 < n; ++k) {
        if (!house_col(H, k + 1, k, v, beta, alpha)) {
            continue;
        }
        apply_left(H, k + 1, v, beta, k);
        H(k + 1, k) = alpha;
        for (size_t i = k + 2; i < n; ++i) {
            H(i, k) = 0.0;
        }
        apply_right(H, k + 1, v, beta, 0);
        apply_right(Q, k + 1, v, beta, 0);
    }
}

// ---------------------------------------------------------------------------
// Real Schur iteration: implicit Francis double-shift QR with deflation
// ---------------------------------------------------------------------------

// One implicit double-shift sweep over the active block [lo, hi] (size >= 3),
// chasing the bulge down the subdiagonal.  s_trace/t_det are the trace and
// determinant of the (possibly complex) shift pair.
void francis_step(Matrix<double>& H, Matrix<double>& Q,
                  size_t lo, size_t hi, double s_trace, double t_det) {
    double x = H(lo, lo) * H(lo, lo) + H(lo, lo + 1) * H(lo + 1, lo)
             - s_trace * H(lo, lo) + t_det;
    double y = H(lo + 1, lo) * (H(lo, lo) + H(lo + 1, lo + 1) - s_trace);
    double z = H(lo + 1, lo) * H(lo + 2, lo + 1);

    std::vector<double> v;
    for (size_t t = lo; t + 2 <= hi; ++t) {
        const double tail_sq = y * y + z * z;
        if (tail_sq != 0.0) {
            const double scale = std::sqrt(x * x + tail_sq);
            const double sign = (x >= 0.0) ? 1.0 : -1.0;
            v.assign(3, 0.0);
            v[0] = x + sign * scale;
            v[1] = y;
            v[2] = z;
            const double vtv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            if (vtv > 0.0) {
                const double beta = 2.0 / vtv;
                apply_left(H, t, v, beta, 0);
                apply_right(H, t, v, beta, 0);
                apply_right(Q, t, v, beta, 0);
            }
        }
        x = H(t + 1, t);
        y = H(t + 2, t);
        z = (t + 3 <= hi) ? H(t + 3, t) : 0.0;
    }

    // Final 2-vector reflector on rows hi-1, hi.
    if (y != 0.0) {
        const double scale = std::sqrt(x * x + y * y);
        const double sign = (x >= 0.0) ? 1.0 : -1.0;
        v.assign(2, 0.0);
        v[0] = x + sign * scale;
        v[1] = y;
        const double vtv = v[0] * v[0] + v[1] * v[1];
        if (vtv > 0.0) {
            const double beta = 2.0 / vtv;
            apply_left(H, hi - 1, v, beta, 0);
            apply_right(H, hi - 1, v, beta, 0);
            apply_right(Q, hi - 1, v, beta, 0);
        }
    }

    // The bulge chase restores the Hessenberg form exactly in exact
    // arithmetic; drop the round-off fill so later rotations cannot smear it
    // back onto the subdiagonal.
    for (size_t j = lo; j + 2 <= hi; ++j) {
        for (size_t i = j + 2; i <= hi; ++i) {
            H(i, j) = 0.0;
        }
    }
}

// Triangularise a converged 2x2 diagonal block when its eigenvalues are real.
// The Givens rotation G has the eigenvector of the larger root as its first
// column, so (G^T B G) e_0 = lambda_1 e_0 and the (1,0) entry becomes zero.
// A block with a complex-conjugate pair is left alone: that is exactly the
// 2x2 block of the real Schur form.
void split_2x2(Matrix<double>& T, Matrix<double>& Q, size_t p, size_t q) {
    const double a = T(p, p);
    const double b = T(p, q);
    const double c = T(q, p);
    const double d = T(q, q);
    if (c == 0.0) {
        return;
    }
    const double half = 0.5 * (a - d);
    const double disc = half * half + b * c;
    if (disc < 0.0) {
        return;  // complex-conjugate pair: keep the 2x2 real Schur block
    }
    double root = std::sqrt(disc);
    root = (half >= 0.0) ? half + root : half - root;

    // Eigenvector of the block for lambda = d + root is [root, c].
    const double nrm = std::hypot(root, c);
    if (nrm == 0.0) {
        return;
    }
    const double cs = root / nrm;
    const double sn = c / nrm;

    const size_t n = T.rows();
    for (size_t j = 0; j < n; ++j) {
        const double t0 = T(p, j);
        const double t1 = T(q, j);
        T(p, j) = cs * t0 + sn * t1;
        T(q, j) = -sn * t0 + cs * t1;
    }
    for (size_t i = 0; i < n; ++i) {
        const double t0 = T(i, p);
        const double t1 = T(i, q);
        T(i, p) = cs * t0 + sn * t1;
        T(i, q) = -sn * t0 + cs * t1;
    }
    for (size_t i = 0; i < n; ++i) {
        const double q0 = Q(i, p);
        const double q1 = Q(i, q);
        Q(i, p) = cs * q0 + sn * q1;
        Q(i, q) = -sn * q0 + cs * q1;
    }
    T(q, p) = 0.0;
}

// Drives the shifted QR iteration on an upper Hessenberg T until every
// diagonal block is 1x1 or a 2x2 with a complex-conjugate pair.
// Returns the number of sweeps used, or ConvergenceFail.
Result<size_t> schur_iterate(Matrix<double>& T, Matrix<double>& Q) {
    const size_t n = T.rows();
    if (n <= 1) {
        return size_t{0};
    }
    const double eps = std::numeric_limits<double>::epsilon();
    const double anorm = frobenius_norm(T);
    const double floor_tol = eps * (anorm > 0.0 ? anorm : 1.0);

    const size_t max_sweeps = 40 * n + 100;
    size_t sweeps = 0;
    size_t block_iter = 0;
    size_t hi = n - 1;

    while (true) {
        size_t lo = hi;
        while (lo > 0) {
            const double s = std::abs(T(lo - 1, lo - 1)) + std::abs(T(lo, lo));
            double thr = eps * s;
            if (thr < floor_tol) {
                thr = floor_tol;
            }
            if (std::abs(T(lo, lo - 1)) <= thr) {
                T(lo, lo - 1) = 0.0;
                break;
            }
            --lo;
        }

        if (lo == hi) {  // 1x1 block converged
            if (hi == 0) {
                break;
            }
            --hi;
            block_iter = 0;
            continue;
        }
        if (lo + 1 == hi) {  // 2x2 block converged
            split_2x2(T, Q, lo, hi);
            if (lo == 0) {
                break;
            }
            hi = lo - 1;
            block_iter = 0;
            continue;
        }

        if (sweeps >= max_sweeps) {
            double worst = 0.0;
            for (size_t i = lo + 1; i <= hi; ++i) {
                worst = std::max(worst, std::abs(T(i, i - 1)));
            }
            return std::unexpected(ConvergenceFail{sweeps, worst});
        }

        double s_trace = 0.0;
        double t_det = 0.0;
        if (block_iter > 0 && block_iter % 10 == 0) {
            // Exceptional (ad-hoc) shift to break a stalled cycle.
            const double s = std::abs(T(hi, hi - 1)) + std::abs(T(hi - 1, hi - 2));
            s_trace = 1.5 * s;
            t_det = s * s;
        } else {
            s_trace = T(hi - 1, hi - 1) + T(hi, hi);
            t_det = T(hi - 1, hi - 1) * T(hi, hi) - T(hi - 1, hi) * T(hi, hi - 1);
        }
        francis_step(T, Q, lo, hi, s_trace, t_det);
        ++sweeps;
        ++block_iter;
    }

    // Everything strictly below the subdiagonal is mathematically zero; clear
    // the accumulated round-off so T is exactly quasi-upper-triangular.
    for (size_t j = 0; j + 2 < n; ++j) {
        for (size_t i = j + 2; i < n; ++i) {
            T(i, j) = 0.0;
        }
    }
    return sweeps;
}

} // namespace

namespace linalg_detail {

// Shared with eig.cpp: real Schur form A = Q*T*Q^T of a square double matrix.
Result<SchurResult> real_schur(const Matrix<double>& A) {
    const size_t n = A.rows();
    Matrix<double> T = copy(A);
    Matrix<double> Q = eye<double>(n);
    if (n == 0) {
        return SchurResult{T, Q};
    }
    hess_reduce(T, Q);
    auto sweeps = schur_iterate(T, Q);
    if (!sweeps) {
        return std::unexpected(sweeps.error());
    }
    return SchurResult{T, Q};
}

} // namespace linalg_detail

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<LdlResult> ldl(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }
    if (!is_symmetric(A)) {
        return std::unexpected(DomainError{"ldl", "matrix not symmetric"});
    }

    const size_t n = A.rows();
    Matrix<double> L = eye<double>(n);
    Matrix<double> D(n, 1, 0.0);
    Matrix<double> work = copy(A);

    std::vector<size_t> perm(n);
    std::iota(perm.begin(), perm.end(), size_t{0});

    double amax = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            amax = std::max(amax, std::abs(work(i, j)));
        }
    }
    const double zero_tol = (amax > 0.0 ? amax : 1.0)
                          * static_cast<double>(n)
                          * std::numeric_limits<double>::epsilon();

    // Threshold symmetric (diagonal) pivoting: the natural pivot is kept
    // unless it has lost roughly half of the available precision relative to
    // the best remaining diagonal, in which case rows/columns are interchanged.
    constexpr double kPivotRatio = 1e-8;

    std::vector<double> cand(n, 0.0);
    for (size_t j = 0; j < n; ++j) {
        double best = 0.0;
        size_t best_i = j;
        for (size_t i = j; i < n; ++i) {
            double di = work(i, i);
            for (size_t k = 0; k < j; ++k) {
                di -= L(i, k) * L(i, k) * D(k, 0);
            }
            cand[i] = di;
            if (std::abs(di) > best) {
                best = std::abs(di);
                best_i = i;
            }
        }

        if (best <= zero_tol) {
            // No usable 1x1 pivot anywhere in the trailing block.
            double off = 0.0;
            for (size_t i = j; i < n; ++i) {
                for (size_t k = i + 1; k < n; ++k) {
                    double v = work(i, k);
                    for (size_t t = 0; t < j; ++t) {
                        v -= L(i, t) * L(k, t) * D(t, 0);
                    }
                    off = std::max(off, std::abs(v));
                }
            }
            if (off <= zero_tol) {
                return std::unexpected(SingularMatrix{});
            }
            return std::unexpected(DomainError{
                "ldl",
                "trailing block has no nonzero diagonal pivot; a 2x2 "
                "Bunch-Kaufman block pivot would be required and only 1x1 "
                "symmetric pivoting is implemented"});
        }

        if (std::abs(cand[j]) < kPivotRatio * best && best_i != j) {
            for (size_t t = 0; t < n; ++t) {
                std::swap(work(j, t), work(best_i, t));
            }
            for (size_t t = 0; t < n; ++t) {
                std::swap(work(t, j), work(t, best_i));
            }
            for (size_t k = 0; k < j; ++k) {
                std::swap(L(j, k), L(best_i, k));
            }
            std::swap(perm[j], perm[best_i]);
            std::swap(cand[j], cand[best_i]);
        }

        D(j, 0) = cand[j];
        for (size_t i = j + 1; i < n; ++i) {
            double lij = work(i, j);
            for (size_t k = 0; k < j; ++k) {
                lij -= L(i, k) * L(j, k) * D(k, 0);
            }
            L(i, j) = lij / D(j, 0);
        }
    }

    Matrix<double> P = zeros<double>(n, n);
    for (size_t j = 0; j < n; ++j) {
        P(perm[j], j) = 1.0;
    }

    return LdlResult{L, D, P};
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> hess(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }

    const size_t n = A.rows();
    Matrix<double> H = copy(A);
    Matrix<double> Q = eye<double>(n);
    hess_reduce(H, Q);
    return H;
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<BidiagResult> bidiag(const Matrix<S, OA, Alloc>& A) {
    const size_t m = A.rows();
    const size_t n = A.cols();
    Matrix<double> U = eye<double>(m);
    Matrix<double> B = copy(A);
    Matrix<double> V = eye<double>(n);

    std::vector<double> v;
    double beta = 0.0;
    double alpha = 0.0;

    const size_t lim = std::min(m, n);
    for (size_t k = 0; k < lim; ++k) {
        // Left reflector spanning rows k..m-1: annihilates the WHOLE
        // subdiagonal of column k (this is the Golub-Kahan step; the
        // Hessenberg variant, which starts one row lower, would leave
        // B(k+1, k) nonzero and produce a tridiagonal B).
        if (house_col(B, k, k, v, beta, alpha)) {
            apply_left(B, k, v, beta, k);
            B(k, k) = alpha;
            for (size_t i = k + 1; i < m; ++i) {
                B(i, k) = 0.0;
            }
            apply_right(U, k, v, beta, 0);
        }
        // Right reflector spanning columns k+1..n-1: annihilates everything
        // to the right of the first superdiagonal in row k.
        if (k + 2 < n && house_row(B, k, k + 1, v, beta, alpha)) {
            apply_right(B, k + 1, v, beta, 0);
            B(k, k + 1) = alpha;
            for (size_t j = k + 2; j < n; ++j) {
                B(k, j) = 0.0;
            }
            apply_right(V, k + 1, v, beta, 0);
        }
    }

    return BidiagResult{U, B, V};
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<SchurResult> schur(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }
    return linalg_detail::real_schur(copy(A));
}

template auto ldl<double>(const Matrix<double>&) -> Result<LdlResult>;
template auto hess<double>(const Matrix<double>&) -> Result<Matrix<double>>;
template auto bidiag<double>(const Matrix<double>&) -> Result<BidiagResult>;
template auto schur<double>(const Matrix<double>&) -> Result<SchurResult>;

} // namespace ms
