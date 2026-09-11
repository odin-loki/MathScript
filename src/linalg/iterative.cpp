// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "ms/linalg/linalg.hpp"
#include "detail.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace ms {

namespace {

using namespace linalg_detail;

Matrix<double> matvec(const Matrix<double>& A, const Matrix<double>& x) {
    return multiply(A, x);
}

double dotvec(const Matrix<double>& a, const Matrix<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.rows(); ++i) {
        sum += a(i, 0) * b(i, 0);
    }
    return sum;
}

Matrix<double> axpy(double alpha, const Matrix<double>& x, const Matrix<double>& y) {
    Matrix<double> r(x.rows(), x.cols());
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            r(i, j) = alpha * x(i, j) + y(i, j);
        }
    }
    return r;
}

// Magnitude below which a scalar is treated as an exact zero (breakdown).
constexpr double kTiny = 1e-30;

// y = A^T * x for a column vector x. A is never transposed into storage.
Matrix<double> matvec_t(const Matrix<double>& A, const Matrix<double>& x) {
    Matrix<double> y(A.cols(), 1, 0.0);
    for (size_t j = 0; j < A.cols(); ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < A.rows(); ++i) {
            sum += A(i, j) * x(i, 0);
        }
        y(j, 0) = sum;
    }
    return y;
}

double norm2(const Matrix<double>& v) {
    return std::sqrt(dotvec(v, v));
}

Matrix<double> scale_vec(double alpha, const Matrix<double>& x) {
    Matrix<double> r(x.rows(), x.cols());
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            r(i, j) = alpha * x(i, j);
        }
    }
    return r;
}

// b - A*x: the true residual, used to confirm quasi-residual estimates.
Matrix<double> residual_vec(const Matrix<double>& A, const Matrix<double>& x,
                            const Matrix<double>& b) {
    return axpy(-1.0, matvec(A, x), b);
}

// Column j of B as an (n, 1) matrix.
Matrix<double> column_of(const Matrix<double>& B, size_t j) {
    Matrix<double> c(B.rows(), 1, 0.0);
    for (size_t i = 0; i < B.rows(); ++i) {
        c(i, 0) = B(i, j);
    }
    return c;
}

// Every Krylov method here works on ONE right-hand side. This runs the given
// single-vector solver once per column of b and assembles the columns of the
// result, so a multi-column b behaves the way solve() does instead of
// returning zeros, stale values, or a silently narrowed matrix.
template<typename Solver>
Result<Matrix<double>> solve_per_column(const Matrix<double>& b, size_t nrows,
                                        Solver&& solver) {
    Matrix<double> X(nrows, b.cols(), 0.0);
    for (size_t j = 0; j < b.cols(); ++j) {
        auto xj = solver(column_of(b, j));
        if (!xj) {
            return std::unexpected(xj.error());
        }
        for (size_t i = 0; i < nrows; ++i) {
            X(i, j) = (*xj)(i, 0);
        }
    }
    return X;
}

// The four solvers that do not go through solve_per_column above (qmr, tfqmr, lsqr, lsmr)
// build their work vectors from b's full shape while every matrix-vector product they take
// produces a single column. A multi-column b therefore made axpy() read past the shorter
// operand: lsmr(A, B) with a 3x3 B overran a three-element buffer (caught by
// AddressSanitizer). They now split b the same way the others do.
template<typename S, StorageOrder OA, template<typename> class Alloc, typename Solver>
Result<Matrix<S, OA, Alloc>> per_column_solve(const Matrix<S, OA, Alloc>& b, size_t out_rows,
                                              Solver&& solve_one) {
    Matrix<S, OA, Alloc> X(out_rows, b.cols(), S(0));
    for (size_t j = 0; j < b.cols(); ++j) {
        Matrix<S, OA, Alloc> rhs(b.rows(), 1);
        for (size_t i = 0; i < b.rows(); ++i) {
            rhs(i, 0) = b(i, j);
        }
        auto xj = solve_one(rhs);
        if (!xj) {
            return std::unexpected(xj.error());
        }
        for (size_t i = 0; i < out_rows; ++i) {
            X(i, j) = (*xj)(i, 0);
        }
    }
    return X;
}

Result<Matrix<double>> cg_single(const Matrix<double>& A, const Matrix<double>& b,
                                 size_t max_iter, double tol) {
    Matrix<double> x(b.rows(), 1, 0.0);
    Matrix<double> r = copy(b);
    Matrix<double> p = copy(r);
    double rsold = dotvec(r, r);
    if (std::sqrt(rsold) < tol) {
        return x;
    }

    for (size_t k = 0; k < max_iter; ++k) {
        Matrix<double> Ap = matvec(A, p);
        const double pAp = dotvec(p, Ap);
        if (std::abs(pAp) < kTiny) {
            return std::unexpected(ConvergenceFail{k, std::sqrt(rsold)});
        }
        const double alpha = rsold / pAp;
        x = axpy(alpha, p, x);
        r = axpy(-alpha, Ap, r);
        const double rsnew = dotvec(r, r);
        if (std::sqrt(rsnew) < tol) {
            return x;
        }
        p = axpy(rsnew / rsold, p, r);
        rsold = rsnew;
    }

    return std::unexpected(ConvergenceFail{max_iter, std::sqrt(rsold)});
}

Result<Matrix<double>> jacobi_single(const Matrix<double>& A, const Matrix<double>& b,
                                     size_t max_iter, double tol) {
    const size_t n = b.rows();
    Matrix<double> x(n, 1, 0.0);
    double res_norm = norm2(b);

    for (size_t k = 0; k < max_iter; ++k) {
        Matrix<double> Ax = matvec(A, x);
        Matrix<double> x_new(n, 1, 0.0);
        for (size_t i = 0; i < n; ++i) {
            if (std::abs(A(i, i)) < kTiny) {
                return std::unexpected(DomainError{"jacobi", "zero diagonal entry"});
            }
            x_new(i, 0) = (b(i, 0) - Ax(i, 0) + A(i, i) * x(i, 0)) / A(i, i);
        }
        res_norm = norm2(residual_vec(A, x_new, b));
        if (res_norm < tol) {
            return x_new;
        }
        x = std::move(x_new);
    }

    return std::unexpected(ConvergenceFail{max_iter, res_norm});
}

Result<Matrix<double>> bicgstab_single(const Matrix<double>& A, const Matrix<double>& b,
                                       size_t max_iter, double tol) {
    const size_t n = b.rows();
    Matrix<double> x(n, 1, 0.0);
    const double bnorm = norm2(b);
    if (bnorm < kTiny) {
        return x;  // b == 0, so x == 0 exactly
    }
    const double target = tol * std::max(1.0, bnorm);

    Matrix<double> r = copy(b);
    Matrix<double> r0 = copy(r);
    Matrix<double> p = copy(r);
    Matrix<double> v(n, 1, 0.0);

    double rho = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    size_t k = 0;
    for (; k < max_iter; ++k) {
        const double rho1 = dotvec(r0, r);
        if (std::abs(rho1) < kTiny) {
            break;  // Lanczos breakdown; the residual check below decides
        }
        if (k == 0) {
            p = copy(r);
        } else {
            if (std::abs(omega) < kTiny) {
                break;
            }
            const double beta = (rho1 / rho) * (alpha / omega);
            Matrix<double> ph = axpy(-omega, v, p);
            p = axpy(beta, ph, r);
        }
        v = matvec(A, p);
        const double r0v = dotvec(r0, v);
        if (std::abs(r0v) < kTiny) {
            break;  // would divide by zero and manufacture NaNs
        }
        alpha = rho1 / r0v;
        Matrix<double> s = axpy(-alpha, v, r);
        if (norm2(s) < target) {
            x = axpy(alpha, p, x);
            break;
        }
        Matrix<double> t = matvec(A, s);
        const double tt = dotvec(t, t);
        if (tt < kTiny) {
            x = axpy(alpha, p, x);
            break;
        }
        omega = dotvec(t, s) / tt;
        x = axpy(alpha, p, axpy(omega, s, x));
        r = axpy(-omega, t, s);
        if (norm2(r) < target) {
            break;
        }
        rho = rho1;
    }

    // The recursively updated r (and a breakdown even more so) is not evidence
    // that x solves the system: confirm against the true residual, and report
    // failure the way cg/jacobi/gmres/minres/qmr in this file do.
    const double res = norm2(residual_vec(A, x, b));
    if (std::isfinite(res) && res <= 10.0 * target) {
        return x;
    }
    return std::unexpected(ConvergenceFail{k, res});
}

Result<Matrix<double>> gmres_single(const Matrix<double>& A, const Matrix<double>& b,
                                    size_t restart, size_t max_iter, double tol) {
    if (restart == 0) {
        restart = 1;
    }
    Matrix<double> x(b.rows(), 1, 0.0);
    Matrix<double> r = copy(b);

    for (size_t outer = 0; outer < max_iter; outer += restart) {
        const double beta = norm2(r);
        if (beta < tol) {
            return x;
        }

        std::vector<Matrix<double>> V(restart + 1);
        V[0] = Matrix<double>(b.rows(), 1, 0.0);
        for (size_t i = 0; i < b.rows(); ++i) {
            V[0](i, 0) = r(i, 0) / beta;
        }

        std::vector<double> g(restart + 1, 0.0);
        g[0] = beta;
        std::vector<std::vector<double>> H(
            restart + 1, std::vector<double>(restart, 0.0));
        std::vector<double> cs(restart, 0.0);
        std::vector<double> sn(restart, 0.0);

        size_t j = 0;  // number of Arnoldi vectors built
        const size_t iter_limit = std::min(restart, max_iter - outer);
        for (size_t step = 0; step < iter_limit; ++step) {
            Matrix<double> w = matvec(A, V[step]);
            for (size_t i = 0; i <= step; ++i) {
                H[i][step] = dotvec(V[i], w);
                w = axpy(-H[i][step], V[i], w);
            }
            H[step + 1][step] = norm2(w);
            const bool invariant = (H[step + 1][step] < 1e-14);
            if (!invariant && step + 1 < restart) {
                V[step + 1] = Matrix<double>(b.rows(), 1, 0.0);
                for (size_t i = 0; i < b.rows(); ++i) {
                    V[step + 1](i, 0) = w(i, 0) / H[step + 1][step];
                }
            }

            for (size_t i = 0; i < step; ++i) {
                const double h0 =  cs[i] * H[i][step] + sn[i] * H[i + 1][step];
                const double h1 = -sn[i] * H[i][step] + cs[i] * H[i + 1][step];
                H[i][step]     = h0;
                H[i + 1][step] = h1;
            }
            const double denom = std::sqrt(H[step][step] * H[step][step] +
                                           H[step + 1][step] * H[step + 1][step]);
            if (denom >= 1e-14) {
                cs[step] = H[step][step] / denom;
                sn[step] = H[step + 1][step] / denom;
                H[step][step]     = denom;
                H[step + 1][step] = 0.0;
                const double g0 = g[step];
                g[step]     =  cs[step] * g0;
                g[step + 1] = -sn[step] * g0;
            }

            j = step + 1;
            if (invariant || std::abs(g[step + 1]) < tol) {
                break;
            }
        }

        Matrix<double> y(j, 1, 0.0);
        for (size_t ii = j; ii-- > 0;) {
            double sum = g[ii];
            for (size_t k = ii + 1; k < j; ++k) {
                sum -= H[ii][k] * y(k, 0);
            }
            if (std::abs(H[ii][ii]) < kTiny) {
                return std::unexpected(ConvergenceFail{outer + j, norm2(r)});
            }
            y(ii, 0) = sum / H[ii][ii];
        }

        for (size_t i = 0; i < j; ++i) {
            x = axpy(y(i, 0), V[i], x);
        }

        r = residual_vec(A, x, b);
        if (norm2(r) < tol) {
            return x;
        }
    }

    return std::unexpected(ConvergenceFail{max_iter, norm2(r)});
}

// MINRES (Paige & Saunders 1975): Lanczos tridiagonalisation of the symmetric
// A, whose tridiagonal least-squares problem is reduced by Givens rotations.
//   oldeps = eps; delta = c*dbar + s*alpha; gbar = s*dbar - c*alpha;
//   eps = s*beta_next; dbar = -c*beta_next; gamma = hypot(gbar, beta_next);
//   c = gbar/gamma; s = beta_next/gamma; phi = c*phibar; phibar = s*phibar;
//   w_new = (v - oldeps*w_prev - delta*w) / gamma;  x += phi*w_new
// phibar is exactly ||b - A x_k||, so it is a genuine residual estimate; it is
// still confirmed against the true residual before x is returned.
Result<Matrix<double>> minres_single(const Matrix<double>& A, const Matrix<double>& b,
                                     size_t max_iter, double tol) {
    const size_t n = b.rows();
    Matrix<double> x(n, 1, 0.0);

    const double beta1 = norm2(b);
    if (beta1 < kTiny) {
        return x;  // b == 0
    }
    const double target = tol * std::max(1.0, beta1);

    Matrix<double> r1 = copy(b);   // v_{j-1} scaled by beta_{j-1}
    Matrix<double> r2 = copy(b);   // v_j scaled by beta_j
    Matrix<double> y = copy(b);

    double oldb = 0.0;
    double beta = beta1;
    double dbar = 0.0;
    double epsln = 0.0;
    double phibar = beta1;
    double cs = -1.0;
    double sn = 0.0;

    Matrix<double> w(n, 1, 0.0);
    Matrix<double> w2(n, 1, 0.0);

    size_t itn = 0;
    for (; itn < max_iter; ++itn) {
        // --- Lanczos step: v = y/beta, y = A*v - (beta/oldb)*r1 - (alfa/beta)*r2
        const double s_inv = 1.0 / beta;
        Matrix<double> v = scale_vec(s_inv, y);
        y = matvec(A, v);
        if (itn >= 1) {
            y = axpy(-(beta / oldb), r1, y);
        }
        const double alfa = dotvec(v, y);
        y = axpy(-(alfa / beta), r2, y);
        r1 = r2;
        r2 = y;
        oldb = beta;
        beta = norm2(r2);

        // --- Apply the previous rotation, then build the next one.
        const double oldeps = epsln;
        const double delta = cs * dbar + sn * alfa;
        const double gbar  = sn * dbar - cs * alfa;
        epsln =  sn * beta;
        dbar  = -cs * beta;

        double gamma = std::hypot(gbar, beta);
        if (gamma < kTiny) {
            gamma = kTiny;
        }
        cs = gbar / gamma;
        sn = beta / gamma;
        const double phi = cs * phibar;
        phibar = std::abs(sn) * phibar;

        // --- Direction update: the whole vector is divided by gamma, and the
        //     delta/oldeps coefficients sit on w (j-1) and w2 (j-2).
        Matrix<double> w1 = w2;
        w2 = w;
        Matrix<double> w_new = axpy(-oldeps, w1, axpy(-delta, w2, v));
        w = scale_vec(1.0 / gamma, w_new);
        x = axpy(phi, w, x);

        if (phibar <= target) {
            ++itn;
            break;
        }
        if (beta < kTiny) {
            ++itn;
            break;  // Krylov space exhausted
        }
    }

    const double res = norm2(residual_vec(A, x, b));
    if (std::isfinite(res) && res <= 10.0 * target) {
        return x;
    }
    return std::unexpected(ConvergenceFail{itn, res});
}


} // namespace

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> cg(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter,
    S tol) {
    if (A.rows() != A.cols() || A.rows() != b.rows()) {
        return std::unexpected(DimensionMismatch{A.rows(), b.rows()});
    }
    if (!is_symmetric(A)) {
        return std::unexpected(DomainError{"cg", "matrix not symmetric"});
    }
    const Matrix<double> Ad = to_col_major(A);
    const Matrix<double> bd = to_col_major(b);
    const double dtol = static_cast<double>(tol);
    return solve_per_column(bd, A.rows(), [&](const Matrix<double>& rhs) {
        return cg_single(Ad, rhs, max_iter, dtol);
    });
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> jacobi(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter,
    S tol) {
    if (A.rows() != A.cols() || A.rows() != b.rows()) {
        return std::unexpected(DimensionMismatch{A.rows(), b.rows()});
    }
    const Matrix<double> Ad = to_col_major(A);
    const Matrix<double> bd = to_col_major(b);
    const double dtol = static_cast<double>(tol);
    return solve_per_column(bd, A.rows(), [&](const Matrix<double>& rhs) {
        return jacobi_single(Ad, rhs, max_iter, dtol);
    });
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> bicgstab(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter,
    S tol) {
    if (A.rows() != A.cols() || A.rows() != b.rows()) {
        return std::unexpected(DimensionMismatch{A.rows(), b.rows()});
    }
    const Matrix<double> Ad = to_col_major(A);
    const Matrix<double> bd = to_col_major(b);
    const double dtol = static_cast<double>(tol);
    return solve_per_column(bd, A.rows(), [&](const Matrix<double>& rhs) {
        return bicgstab_single(Ad, rhs, max_iter, dtol);
    });
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> gmres(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t restart,
    size_t max_iter,
    S tol) {
    if (A.rows() != A.cols() || A.rows() != b.rows()) {
        return std::unexpected(DimensionMismatch{A.rows(), b.rows()});
    }
    const Matrix<double> Ad = to_col_major(A);
    const Matrix<double> bd = to_col_major(b);
    const double dtol = static_cast<double>(tol);
    return solve_per_column(bd, A.rows(), [&](const Matrix<double>& rhs) {
        return gmres_single(Ad, rhs, restart, max_iter, dtol);
    });
}

template auto cg<double>(const Matrix<double>&, const Matrix<double>&, size_t, double)
    -> Result<Matrix<double>>;
template auto jacobi<double>(const Matrix<double>&, const Matrix<double>&, size_t, double)
    -> Result<Matrix<double>>;
template auto bicgstab<double>(const Matrix<double>&, const Matrix<double>&, size_t, double)
    -> Result<Matrix<double>>;
template auto gmres<double>(const Matrix<double>&, const Matrix<double>&, size_t, size_t, double)
    -> Result<Matrix<double>>;

// --- MINRES ---
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> minres(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter,
    S tol) {
    if (A.rows() != A.cols() || A.rows() != b.rows()) {
        return std::unexpected(DimensionMismatch{A.rows(), b.rows()});
    }
    const Matrix<double> Ad = to_col_major(A);
    const Matrix<double> bd = to_col_major(b);
    const double dtol = static_cast<double>(tol);
    return solve_per_column(bd, A.rows(), [&](const Matrix<double>& rhs) {
        return minres_single(Ad, rhs, max_iter, dtol);
    });
}

template auto minres<double>(const Matrix<double>&, const Matrix<double>&, size_t, double)
    -> Result<Matrix<double>>;

// --- QMR (Quasi-Minimal Residual, Freund & Nachtigal 1991) ---
// Unsymmetric (two-sided) Lanczos with A and A^T builds the biorthogonal
// bases; the resulting tridiagonal least-squares problem is smoothed by
// Givens rotations, which is what makes the residual curve monotone-ish
// where BiCG's oscillates. No look-ahead: a serious Lanczos breakdown ends
// the iteration and is reported instead of being stepped over.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> qmr(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter,
    S tol) {
    if (A.rows() != A.cols() || A.rows() != b.rows()) {
        return std::unexpected(DimensionMismatch{A.rows(), b.rows()});
    }
    if (b.cols() != 1) {
        return per_column_solve(b, b.rows(), [&](const Matrix<S, OA, Alloc>& rhs) {
            return qmr(A, rhs, max_iter, tol);
        });
    }
    const size_t n = b.rows();
    Matrix<double> x(n, 1, 0.0);

    const double norm_b = norm2(b);
    if (norm_b < kTiny) {
        return x;
    }

    // x_0 = 0, so r_0 = b. Both starting Lanczos vectors are r_0.
    Matrix<double> r = copy(b);
    Matrix<double> v_tilde = copy(r);
    Matrix<double> w_tilde = copy(r);
    double rho = norm2(v_tilde);
    double xi = norm2(w_tilde);

    double gamma = 1.0;
    double eta = -1.0;
    double theta = 0.0;
    double epsilon = 1.0;

    Matrix<double> p(n, 1, 0.0);
    Matrix<double> q(n, 1, 0.0);
    Matrix<double> d(n, 1, 0.0);
    Matrix<double> s(n, 1, 0.0);

    size_t used = 0;
    for (size_t k = 1; k <= max_iter; ++k) {
        used = k;
        if (std::abs(rho) < kTiny || std::abs(xi) < kTiny) {
            break;  // Lanczos vectors have died out.
        }
        const Matrix<double> v = scale_vec(1.0 / rho, v_tilde);
        const Matrix<double> w = scale_vec(1.0 / xi, w_tilde);

        const double delta = dotvec(w, v);
        if (std::abs(delta) < kTiny) {
            break;  // Serious breakdown; look-ahead would be needed here.
        }
        if (k == 1) {
            p = copy(v);
            q = copy(w);
        } else {
            p = axpy(-(xi * delta / epsilon), p, v);
            q = axpy(-(rho * delta / epsilon), q, w);
        }

        const Matrix<double> p_tilde = matvec(A, p);
        epsilon = dotvec(q, p_tilde);
        if (std::abs(epsilon) < kTiny) {
            break;
        }
        const double beta = epsilon / delta;
        if (std::abs(beta) < kTiny) {
            break;
        }

        v_tilde = axpy(-beta, v, p_tilde);
        const double rho_next = norm2(v_tilde);
        w_tilde = axpy(-beta, w, matvec_t(A, q));
        const double xi_next = norm2(w_tilde);

        // Givens rotation of the new tridiagonal column. theta uses the NEW
        // rho and the PREVIOUS gamma; eta uses the OLD rho, so rho is only
        // shifted after eta has been formed.
        const double theta_prev = theta;
        const double gamma_prev = gamma;
        theta = rho_next / (gamma_prev * std::abs(beta));
        gamma = 1.0 / std::sqrt(1.0 + theta * theta);
        if (std::abs(gamma) < kTiny) {
            break;
        }
        eta = -eta * rho * gamma * gamma / (beta * gamma_prev * gamma_prev);

        if (k == 1) {
            d = scale_vec(eta, p);
            s = scale_vec(eta, p_tilde);
        } else {
            const double c = (theta_prev * gamma) * (theta_prev * gamma);
            d = axpy(c, d, scale_vec(eta, p));
            s = axpy(c, s, scale_vec(eta, p_tilde));
        }
        x = axpy(1.0, d, x);
        r = axpy(-1.0, s, r);  // s == A*d in exact arithmetic

        rho = rho_next;
        xi = xi_next;

        // The recursive r drifts from the true residual, so confirm before
        // declaring success.
        if (norm2(r) <= tol * norm_b) {
            const double true_res = norm2(residual_vec(A, x, b));
            if (true_res <= tol * norm_b) {
                return x;
            }
        }
    }

    const double res_norm = norm2(residual_vec(A, x, b));
    if (res_norm <= 10.0 * tol * norm_b) {
        return x;
    }
    return std::unexpected(ConvergenceFail{used, res_norm});
}

// --- LSQR (Paige & Saunders, least-squares) ---
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> lsqr(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter,
    S tol) {
    if (A.rows() != b.rows()) {
        return std::unexpected(DimensionMismatch{A.rows(), b.rows()});
    }
    if (b.cols() != 1) {
        return per_column_solve(b, A.cols(), [&](const Matrix<S, OA, Alloc>& rhs) {
            return lsqr(A, rhs, max_iter, tol);
        });
    }
    const size_t m = A.rows(), n_cols = A.cols();
    Matrix<double> x(n_cols, 1, 0.0);
    Matrix<double> u = copy(b);
    double beta = std::sqrt(dotvec(u, u));
    if (beta < 1e-30) return x;
    for (size_t i = 0; i < m; ++i) u(i, 0) /= beta;

    // v = A^T u
    Matrix<double> v(n_cols, 1, 0.0);
    for (size_t j = 0; j < n_cols; ++j)
        for (size_t i = 0; i < m; ++i)
            v(j, 0) += A(i, j) * u(i, 0);
    double alpha = std::sqrt(dotvec(v, v));
    if (alpha < 1e-30) return x;
    for (size_t j = 0; j < n_cols; ++j) v(j, 0) /= alpha;

    Matrix<double> w = copy(v);
    double phibar = beta, rhobar = alpha;
    double norm_b = beta;

    for (size_t k = 0; k < max_iter; ++k) {
        // Bidiagonalisation step
        Matrix<double> Au = matvec(A, v);
        u = axpy(-alpha, u, Au);
        beta = std::sqrt(dotvec(u, u));
        if (beta > 1e-30)
            for (size_t i = 0; i < m; ++i) u(i, 0) /= beta;

        Matrix<double> Atb(n_cols, 1, 0.0);
        for (size_t j = 0; j < n_cols; ++j)
            for (size_t i = 0; i < m; ++i)
                Atb(j, 0) += A(i, j) * u(i, 0);
        v = axpy(-beta, v, Atb);
        alpha = std::sqrt(dotvec(v, v));
        if (alpha > 1e-30)
            for (size_t j = 0; j < n_cols; ++j) v(j, 0) /= alpha;

        // Givens rotation
        double rho   = std::sqrt(rhobar * rhobar + beta * beta);
        double c     = rhobar / rho;
        double s     = beta / rho;
        double theta = s * alpha;
        rhobar       = -c * alpha;
        double phi   = c * phibar;
        phibar       = s * phibar;

        x = axpy(phi / rho, w, x);
        w = axpy(-theta / rho, w, v);

        if (std::abs(phibar) / norm_b < tol) return x;
    }
    return x;
}

// --- LSMR (Fong & Saunders 2011) ---
// Golub-Kahan bidiagonalisation, as in lsqr(), but with a second sequence of
// rotations applied to R^T. LSQR minimises ||r|| over the Krylov space; LSMR
// minimises ||A^T r|| over the same space, monotonically. Undamped
// (lambda = 0). Accepts any m x n A and never reports ConvergenceFail: the
// current iterate is always the best approximation found so far.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> lsmr(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter,
    S tol) {
    if (A.rows() != b.rows()) {
        return std::unexpected(DimensionMismatch{A.rows(), b.rows()});
    }
    if (b.cols() != 1) {
        return per_column_solve(b, A.cols(), [&](const Matrix<S, OA, Alloc>& rhs) {
            return lsmr(A, rhs, max_iter, tol);
        });
    }
    const size_t n_cols = A.cols();
    Matrix<double> x(n_cols, 1, 0.0);

    double beta = norm2(b);
    if (beta < kTiny) {
        return x;
    }
    const double norm_b = beta;
    Matrix<double> u = scale_vec(1.0 / beta, b);
    Matrix<double> v = matvec_t(A, u);
    double alpha = norm2(v);
    if (alpha < kTiny) {
        return x;  // b is orthogonal to range(A): x = 0 already solves it.
    }
    v = scale_vec(1.0 / alpha, v);

    double alphabar = alpha;
    double zetabar = alpha * beta;          // |zetabar| tracks ||A^T r||
    const double norm_ar0 = std::abs(zetabar);
    double rho = 1.0;
    double rhobar = 1.0;
    double cbar = 1.0;
    double sbar = 0.0;
    double zeta = 0.0;

    Matrix<double> h = copy(v);
    Matrix<double> hbar(n_cols, 1, 0.0);

    // State for the ||r|| estimate (Fong & Saunders section 3.3).
    double betadd = beta;
    double betad = 0.0;
    double rhodold = 1.0;
    double tautildeold = 0.0;
    double thetatilde = 0.0;
    double norm_r = beta;

    for (size_t k = 0; k < max_iter; ++k) {
        // Bidiagonalisation step.
        u = axpy(-alpha, u, matvec(A, v));
        beta = norm2(u);
        if (beta > kTiny) {
            u = scale_vec(1.0 / beta, u);
            v = axpy(-beta, v, matvec_t(A, u));
            alpha = norm2(v);
            if (alpha > kTiny) {
                v = scale_vec(1.0 / alpha, v);
            } else {
                alpha = 0.0;
            }
        } else {
            beta = 0.0;   // the bidiagonalisation has terminated
            alpha = 0.0;
        }

        // Rotation P_k turns B_k into R_k.
        const double rhoold = rho;
        rho = std::sqrt(alphabar * alphabar + beta * beta);
        if (rho < kTiny) {
            break;
        }
        const double c = alphabar / rho;
        const double s = beta / rho;
        const double thetanew = s * alpha;
        alphabar = c * alpha;

        // Rotation Pbar_k turns R_k^T into Rbar_k.
        const double rhobarold = rhobar;
        const double zetaold = zeta;
        const double thetabar = sbar * rho;
        const double rhotemp = cbar * rho;
        rhobar = std::sqrt(rhotemp * rhotemp + thetanew * thetanew);
        if (rhobar < kTiny) {
            break;
        }
        cbar = rhotemp / rhobar;
        sbar = thetanew / rhobar;
        zeta = cbar * zetabar;
        zetabar = -sbar * zetabar;

        // Update hbar, x, h.
        hbar = axpy(-(thetabar * rho / (rhoold * rhobarold)), hbar, h);
        x = axpy(zeta / (rho * rhobar), hbar, x);
        h = axpy(-(thetanew / rho), h, v);

        // ||r|| estimate. The general formula carries a term d = sum of
        // betacheck^2 from the extra rotation used when a damping parameter
        // lambda is present; undamped, betacheck is identically zero, so the
        // term is omitted rather than missing.
        const double betahat = c * betadd;
        betadd = -s * betadd;

        const double thetatildeold = thetatilde;
        const double rhotildeold =
            std::sqrt(rhodold * rhodold + thetabar * thetabar);
        const double ctildeold = rhodold / rhotildeold;
        const double stildeold = thetabar / rhotildeold;
        thetatilde = stildeold * rhobar;
        rhodold = ctildeold * rhobar;
        betad = -stildeold * betad + ctildeold * betahat;

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold;
        const double taud = (zeta - thetatilde * tautildeold) / rhodold;
        norm_r = std::sqrt((betad - taud) * (betad - taud) + betadd * betadd);

        if (std::abs(zetabar) <= tol * norm_ar0) {
            break;  // least-squares (normal-equation) convergence
        }
        if (norm_r <= tol * norm_b) {
            break;  // consistent-system convergence
        }
    }

    return x;
}

// --- TFQMR (Transpose-Free QMR, Freund 1993) ---
// The CGS recurrence smoothed by QMR's rotations, so only products with A are
// needed. Each outer iteration is two matrix-vector products and two "half
// steps". tau is a quasi-residual bounded by ||b - A x_m|| <= tau*sqrt(m+1),
// so it screens rather than decides: x is updated BEFORE the test (on A = I
// the very half step that first makes x correct is the one that drives tau to
// zero, which is what made the historical loop return x = 0) and the true
// residual is what actually declares convergence.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> tfqmr(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter,
    S tol) {
    if (A.rows() != A.cols() || A.rows() != b.rows()) {
        return std::unexpected(DimensionMismatch{A.rows(), b.rows()});
    }
    if (b.cols() != 1) {
        return per_column_solve(b, b.rows(), [&](const Matrix<S, OA, Alloc>& rhs) {
            return tfqmr(A, rhs, max_iter, tol);
        });
    }
    const size_t n = b.rows();
    Matrix<double> x(n, 1, 0.0);

    const double norm_b = norm2(b);
    if (norm_b < kTiny) {
        return x;
    }

    Matrix<double> r = copy(b);
    Matrix<double> w = copy(r);
    Matrix<double> y_odd = copy(r);
    const Matrix<double> r_shadow = copy(r);
    // u_m = A*y_m; the odd one is carried over from the previous outer step
    // (or from the initialisation) so it is never recomputed.
    Matrix<double> u_odd = matvec(A, y_odd);
    Matrix<double> v = copy(u_odd);
    Matrix<double> d(n, 1, 0.0);
    Matrix<double> y_even(n, 1, 0.0);
    Matrix<double> u_even(n, 1, 0.0);

    double tau = norm2(r);
    double theta = 0.0;
    double eta = 0.0;
    double rho = dotvec(r_shadow, r);
    size_t m = 0;  // half-step counter

    for (size_t it = 1; it <= max_iter; ++it) {
        if (std::abs(rho) < kTiny) {
            break;
        }
        const double sigma = dotvec(r_shadow, v);
        if (std::abs(sigma) < kTiny) {
            break;
        }
        const double alpha = rho / sigma;
        if (std::abs(alpha) < kTiny || !std::isfinite(alpha)) {
            break;
        }

        y_even = axpy(-alpha, v, y_odd);
        u_even = matvec(A, y_even);

        bool converged = false;
        bool stalled = false;
        for (size_t half = 0; half < 2; ++half) {
            const Matrix<double>& y_m = (half == 0) ? y_odd : y_even;
            const Matrix<double>& u_m = (half == 0) ? u_odd : u_even;
            ++m;
            w = axpy(-alpha, u_m, w);

            if (tau < kTiny || !std::isfinite(tau)) {
                stalled = true;
                break;
            }
            const double theta_prev = theta;
            const double eta_prev = eta;
            theta = norm2(w) / tau;
            const double c = 1.0 / std::sqrt(1.0 + theta * theta);
            tau = tau * theta * c;
            eta = c * c * alpha;

            const double dcoef = theta_prev * theta_prev * eta_prev / alpha;
            d = axpy(dcoef, d, y_m);
            x = axpy(eta, d, x);

            if (tau <= tol * norm_b) {
                const double true_res = norm2(residual_vec(A, x, b));
                if (true_res <= tol * norm_b) {
                    converged = true;
                    break;
                }
                if (tau < kTiny) {
                    // The quasi-residual has collapsed with the true residual
                    // still too large: no progress is left in this recurrence.
                    stalled = true;
                    break;
                }
            }
        }
        if (converged) {
            return x;
        }
        if (stalled) {
            break;
        }

        const double rho_next = dotvec(r_shadow, w);
        const double beta = rho_next / rho;
        rho = rho_next;
        y_odd = axpy(beta, y_even, w);
        u_odd = matvec(A, y_odd);
        v = axpy(beta, axpy(beta, v, u_even), u_odd);
    }

    const double res_norm = norm2(residual_vec(A, x, b));
    if (res_norm <= 10.0 * tol * norm_b) {
        return x;
    }
    return std::unexpected(ConvergenceFail{m, res_norm});
}

// --- Preconditioners ---
template<typename S, StorageOrder OA, template<typename> class Alloc>
std::vector<S> precond_diag(const Matrix<S, OA, Alloc>& A) {
    size_t n = std::min(A.rows(), A.cols());
    std::vector<S> d(n);
    for (size_t i = 0; i < n; ++i)
        d[i] = (std::abs(A(i, i)) > 1e-30) ? S(1) / A(i, i) : S(1);
    return d;
}

// M = (D/omega + L) * (D/omega)^-1 * (D/omega + U), formed explicitly.
// The usual 1/(omega*(2 - omega)) normalisation is deliberately omitted: it is
// singular at omega = 2, and a preconditioner is only defined up to a positive
// scalar (the constant cancels out of a Krylov method's alpha and beta).
// Only k <= min(i, j) contributes to entry (i, j), since L is strictly lower
// and U strictly upper.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, OA, Alloc> precond_ssor(const Matrix<S, OA, Alloc>& A, S omega) {
    const size_t n = std::min(A.rows(), A.cols());
    const S w = (std::abs(omega) > S(kTiny)) ? omega : S(1);

    Matrix<S, OA, Alloc> M(n, n, S(0));
    std::vector<S> dw(n, S(0));
    std::vector<S> dinv(n, S(0));
    for (size_t i = 0; i < n; ++i) {
        dw[i] = A(i, i) / w;
        dinv[i] = (std::abs(A(i, i)) > S(kTiny)) ? (w / A(i, i)) : S(0);
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            S sum = S(0);
            const size_t kmax = std::min(i, j);
            for (size_t k = 0; k <= kmax; ++k) {
                const S lik = (i == k) ? dw[i] : A(i, k);
                const S ukj = (k == j) ? dw[j] : A(k, j);
                sum += lik * dinv[k] * ukj;
            }
            M(i, j) = sum;
        }
    }
    return M;
}

// z = M^-1 r for the same M, by forward substitution, diagonal scaling and
// back substitution — M is never formed.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, OA, Alloc> precond_ssor_apply(
    const Matrix<S, OA, Alloc>& A, S omega, const Matrix<S, OA, Alloc>& r) {
    const size_t n = std::min(A.rows(), A.cols());
    Matrix<S, OA, Alloc> z(n, 1, S(0));
    if (r.rows() != n || r.cols() == 0) {
        return z;
    }
    const S w = (std::abs(omega) > S(kTiny)) ? omega : S(1);

    std::vector<S> y(n, S(0));
    for (size_t i = 0; i < n; ++i) {          // (D/omega + L) y = r
        S sum = r(i, 0);
        for (size_t j = 0; j < i; ++j) {
            sum -= A(i, j) * y[j];
        }
        const S dii = A(i, i) / w;
        y[i] = (std::abs(dii) > S(kTiny)) ? (sum / dii) : sum;
    }
    for (size_t i = 0; i < n; ++i) {          // y <- (D/omega) y
        y[i] *= A(i, i) / w;
    }
    for (size_t i = n; i-- > 0;) {            // (D/omega + U) z = y
        S sum = y[i];
        for (size_t j = i + 1; j < n; ++j) {
            sum -= A(i, j) * z(j, 0);
        }
        const S dii = A(i, i) / w;
        z(i, 0) = (std::abs(dii) > S(kTiny)) ? (sum / dii) : sum;
    }
    return z;
}

// ILU(0): IKJ Gaussian elimination restricted to the sparsity pattern of A, so
// no fill-in is created. L (unit diagonal implied) goes in the strictly lower
// triangle, U in the upper triangle including the diagonal. The == S(0) tests
// are deliberate structural comparisons against the pattern of the ORIGINAL A,
// not floating-point tolerance tests.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, OA, Alloc> precond_ilu0(const Matrix<S, OA, Alloc>& A) {
    const size_t n = std::min(A.rows(), A.cols());
    Matrix<S, OA, Alloc> LU(n, n, S(0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            LU(i, j) = A(i, j);
        }
    }
    for (size_t i = 1; i < n; ++i) {
        for (size_t k = 0; k < i; ++k) {
            if (A(i, k) == S(0)) {
                continue;                     // outside pattern(A)
            }
            if (std::abs(LU(k, k)) < S(kTiny)) {
                continue;                     // zero pivot: skip this update
            }
            LU(i, k) /= LU(k, k);
            for (size_t j = k + 1; j < n; ++j) {
                if (A(i, j) == S(0)) {
                    continue;                 // no fill-in permitted
                }
                LU(i, j) -= LU(i, k) * LU(k, j);
            }
        }
    }
    return LU;
}

// Solve (L*U) z = r for the compact factors precond_ilu0() returns.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, OA, Alloc> precond_ilu0_apply(
    const Matrix<S, OA, Alloc>& LU, const Matrix<S, OA, Alloc>& r) {
    const size_t n = std::min(LU.rows(), LU.cols());
    Matrix<S, OA, Alloc> z(n, 1, S(0));
    if (r.rows() != n || r.cols() == 0) {
        return z;
    }
    std::vector<S> y(n, S(0));
    for (size_t i = 0; i < n; ++i) {          // L y = r, L has a unit diagonal
        S sum = r(i, 0);
        for (size_t j = 0; j < i; ++j) {
            sum -= LU(i, j) * y[j];
        }
        y[i] = sum;
    }
    for (size_t i = n; i-- > 0;) {            // U z = y
        S sum = y[i];
        for (size_t j = i + 1; j < n; ++j) {
            sum -= LU(i, j) * z(j, 0);
        }
        z(i, 0) = (std::abs(LU(i, i)) > S(kTiny)) ? (sum / LU(i, i)) : sum;
    }
    return z;
}

// --- Preconditioned conjugate gradient ---
// Textbook PCG with the preconditioner supplied as an operator z = M^-1 r, so
// M never has to be materialised. The stopping test is the absolute residual
// used by cg(), so pcg(A, b, identity, ...) reproduces cg(A, b, ...) exactly.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> pcg(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    const std::function<Matrix<S, OA, Alloc>(const Matrix<S, OA, Alloc>&)>& M_apply,
    size_t max_iter,
    S tol) {
    if (A.rows() != A.cols() || A.rows() != b.rows()) {
        return std::unexpected(DimensionMismatch{A.rows(), b.rows()});
    }
    if (!is_symmetric(A)) {
        return std::unexpected(DomainError{"pcg", "matrix not symmetric"});
    }
    if (!M_apply) {
        return std::unexpected(DomainError{"pcg", "null preconditioner"});
    }

    const size_t n = b.rows();
    Matrix<double> x(n, 1, 0.0);
    Matrix<double> r = copy(b);
    double res = norm2(r);
    if (res < tol) {
        return x;
    }

    Matrix<double> z = M_apply(r);
    if (z.rows() != n || z.cols() == 0) {
        return std::unexpected(
            DomainError{"pcg", "preconditioner returned wrong shape"});
    }
    Matrix<double> p = copy(z);
    double rz = dotvec(r, z);
    if (rz <= 0.0) {
        return std::unexpected(
            DomainError{"pcg", "preconditioner not positive definite"});
    }

    for (size_t k = 0; k < max_iter; ++k) {
        const Matrix<double> Ap = matvec(A, p);
        const double pAp = dotvec(p, Ap);
        if (std::abs(pAp) < kTiny) {
            break;
        }
        const double alpha = rz / pAp;
        x = axpy(alpha, p, x);
        r = axpy(-alpha, Ap, r);
        res = norm2(r);
        if (res < tol) {
            return x;
        }
        z = M_apply(r);
        if (z.rows() != n || z.cols() == 0) {
            return std::unexpected(
                DomainError{"pcg", "preconditioner returned wrong shape"});
        }
        const double rz_next = dotvec(r, z);
        if (rz_next <= 0.0) {
            return std::unexpected(
                DomainError{"pcg", "preconditioner not positive definite"});
        }
        const double beta = rz_next / rz;
        p = axpy(beta, p, z);
        rz = rz_next;
    }

    return std::unexpected(ConvergenceFail{max_iter, res});
}

template auto qmr<double>(const Matrix<double>&, const Matrix<double>&, size_t, double)
    -> Result<Matrix<double>>;
template auto lsqr<double>(const Matrix<double>&, const Matrix<double>&, size_t, double)
    -> Result<Matrix<double>>;
template auto lsmr<double>(const Matrix<double>&, const Matrix<double>&, size_t, double)
    -> Result<Matrix<double>>;
template auto tfqmr<double>(const Matrix<double>&, const Matrix<double>&, size_t, double)
    -> Result<Matrix<double>>;
template auto precond_diag<double>(const Matrix<double>&) -> std::vector<double>;
template auto precond_ssor<double>(const Matrix<double>&, double) -> Matrix<double>;
template auto precond_ssor_apply<double>(const Matrix<double>&, double, const Matrix<double>&)
    -> Matrix<double>;
template auto precond_ilu0<double>(const Matrix<double>&) -> Matrix<double>;
template auto precond_ilu0_apply<double>(const Matrix<double>&, const Matrix<double>&)
    -> Matrix<double>;
template auto pcg<double>(
    const Matrix<double>&, const Matrix<double>&,
    const std::function<Matrix<double>(const Matrix<double>&)>&, size_t, double)
    -> Result<Matrix<double>>;

} // namespace ms
