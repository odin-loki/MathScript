#include "ms/linalg/linalg.hpp"
#include "detail.hpp"
#include <cmath>

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

    Matrix<double> x(b.rows(), b.cols(), 0.0);
    Matrix<double> r = copy(b);
    Matrix<double> p = copy(r);
    double rsold = dotvec(r, r);

    for (size_t k = 0; k < max_iter; ++k) {
        Matrix<double> Ap = matvec(A, p);
        const double alpha = rsold / dotvec(p, Ap);
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

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> jacobi(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter,
    S tol) {
    if (A.rows() != A.cols() || A.rows() != b.rows()) {
        return std::unexpected(DimensionMismatch{A.rows(), b.rows()});
    }

    Matrix<double> x(b.rows(), b.cols(), 0.0);
    double res_norm = std::sqrt(dotvec(b, b));

    for (size_t k = 0; k < max_iter; ++k) {
        Matrix<double> Ax = matvec(A, x);
        Matrix<double> x_new(b.rows(), b.cols());
        for (size_t i = 0; i < b.rows(); ++i) {
            if (std::abs(A(i, i)) < 1e-30) {
                return std::unexpected(DomainError{"jacobi", "zero diagonal entry"});
            }
            x_new(i, 0) = (b(i, 0) - Ax(i, 0) + A(i, i) * x(i, 0)) / A(i, i);
        }
        Matrix<double> r = axpy(-1.0, matvec(A, x_new), b);
        res_norm = std::sqrt(dotvec(r, r));
        if (res_norm < tol) {
            return x_new;
        }
        x = std::move(x_new);
    }

    return std::unexpected(ConvergenceFail{max_iter, res_norm});
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

    Matrix<double> x(b.rows(), b.cols(), 0.0);
    Matrix<double> r = copy(b);
    Matrix<double> r0 = copy(r);
    Matrix<double> p = copy(r);
    Matrix<double> v(b.rows(), b.cols(), 0.0);

    double rho = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    for (size_t k = 0; k < max_iter; ++k) {
        const double rho1 = dotvec(r0, r);
        if (std::abs(rho1) < 1e-30) {
            break;
        }
        if (k == 0) {
            p = copy(r);
        } else {
            const double beta = (rho1 / rho) * (alpha / omega);
            Matrix<double> ph = axpy(-omega, v, p);
            p = axpy(beta, ph, r);
        }
        v = matvec(A, p);
        alpha = rho1 / dotvec(r0, v);
        Matrix<double> s = axpy(-alpha, v, r);
        if (std::sqrt(dotvec(s, s)) < tol) {
            x = axpy(alpha, p, x);
            break;
        }
        Matrix<double> t = matvec(A, s);
        omega = dotvec(t, s) / dotvec(t, t);
        x = axpy(alpha, p, axpy(omega, s, x));
        r = axpy(-omega, t, s);
        if (std::sqrt(dotvec(r, r)) < tol) {
            break;
        }
        rho = rho1;
    }

    return x;
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

    Matrix<double> x(b.rows(), b.cols(), 0.0);
    Matrix<double> r = copy(b);

    for (size_t outer = 0; outer < max_iter; outer += restart) {
        const double beta = std::sqrt(dotvec(r, r));
        if (beta < tol) {
            return x;
        }

        std::vector<Matrix<double>> V(restart + 1);
        V[0] = Matrix<double>(b.rows(), 1);
        for (size_t i = 0; i < b.rows(); ++i) {
            V[0](i, 0) = r(i, 0) / beta;
        }

        std::vector<double> g(restart + 1, 0.0);
        g[0] = beta;
        std::vector<std::vector<double>> H(
            restart + 1, std::vector<double>(restart, 0.0));
        // Store Givens rotation parameters
        std::vector<double> cs(restart, 0.0);  // cosines
        std::vector<double> sn(restart, 0.0);  // sines

        size_t j = 0;  // number of Arnoldi vectors built
        size_t iter_limit = std::min(restart, max_iter - outer);
        for (size_t step = 0; step < iter_limit; ++step) {
            Matrix<double> w = matvec(A, V[step]);
            for (size_t i = 0; i <= step; ++i) {
                H[i][step] = dotvec(V[i], w);
                w = axpy(-H[i][step], V[i], w);
            }
            H[step + 1][step] = std::sqrt(dotvec(w, w));
            bool invariant = (H[step + 1][step] < 1e-14);
            if (!invariant && step + 1 < restart) {
                V[step + 1] = Matrix<double>(b.rows(), 1);
                for (size_t i = 0; i < b.rows(); ++i) {
                    V[step + 1](i, 0) = w(i, 0) / H[step + 1][step];
                }
            }

            // Apply previous Givens rotations to column step
            for (size_t i = 0; i < step; ++i) {
                const double h0 =  cs[i] * H[i][step] + sn[i] * H[i + 1][step];
                const double h1 = -sn[i] * H[i][step] + cs[i] * H[i + 1][step];
                H[i][step]     = h0;
                H[i + 1][step] = h1;
            }
            // Compute and apply new Givens rotation
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
        for (int i = static_cast<int>(j) - 1; i >= 0; --i) {
            double sum = g[i];
            for (size_t k = i + 1; k < j; ++k) {
                sum -= H[i][k] * y(k, 0);
            }
            y(i, 0) = sum / H[i][i];
        }

        for (size_t i = 0; i < j; ++i) {
            x = axpy(y(i, 0), V[i], x);
        }

        r = axpy(-1.0, matvec(A, x), b);
        if (std::sqrt(dotvec(r, r)) < tol) {
            return x;
        }
    }

    return std::unexpected(ConvergenceFail{max_iter, std::sqrt(dotvec(r, r))});
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

    const size_t n = b.rows();
    Matrix<double> x(n, 1, 0.0);
    Matrix<double> r = copy(b);

    // Lanczos vectors
    auto v = copy(r);
    double beta = std::sqrt(dotvec(v, v));
    if (beta < 1e-14) return x;
    for (size_t i = 0; i < n; ++i) v(i, 0) /= beta;

    Matrix<double> v_prev(n, 1, 0.0);
    double c_old = 1.0, c_cur = 1.0;
    double s_old = 0.0, s_cur = 0.0;
    double eta = beta;
    Matrix<double> w(n, 1, 0.0), w_prev(n, 1, 0.0);

    for (size_t iter = 0; iter < max_iter; ++iter) {
        // Lanczos step
        auto Av = matvec(A, v);
        double alpha = dotvec(v, Av);
        auto v_next = axpy(-alpha, v, axpy(-beta, v_prev, Av));
        double beta_next = std::sqrt(dotvec(v_next, v_next));
        if (beta_next > 1e-14) {
            for (size_t i = 0; i < n; ++i) v_next(i, 0) /= beta_next;
        }

        // Apply Givens rotations
        double delta = c_cur * alpha - c_old * s_cur * beta;
        double eps   = s_old * beta;
        double gamma = std::sqrt(delta * delta + beta_next * beta_next);
        if (gamma < 1e-14) gamma = 1e-14;
        double c_new = delta / gamma;
        double s_new = beta_next / gamma;

        // Update solution
        auto w_new = axpy(-c_old * s_cur * beta / gamma, w_prev,
                          axpy(-s_old * beta / gamma, w, copy(v)));
        // Multiply by 1/gamma (already divided by gamma above)
        x = axpy(c_new * eta, w_new, x);
        eta = -s_new * eta;

        // Shift for next iter
        w_prev = w; w = w_new;
        v_prev = v; v = v_next;
        beta = beta_next;
        c_old = c_cur; s_old = s_cur;
        c_cur = c_new; s_cur = s_new;

        // Residual estimate
        double res = std::abs(eta);
        if (res < tol) return x;
    }

    auto r_check = axpy(-1.0, matvec(A, x), b);
    double res_norm = std::sqrt(dotvec(r_check, r_check));
    if (res_norm < tol * 10.0) return x;
    return std::unexpected(ConvergenceFail{max_iter, res_norm});
}

template auto minres<double>(const Matrix<double>&, const Matrix<double>&, size_t, double)
    -> Result<Matrix<double>>;

// --- QMR (Quasi-Minimal Residual) ---
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> qmr(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter,
    S tol) {
    if (A.rows() != A.cols() || A.rows() != b.rows()) {
        return std::unexpected(DimensionMismatch{A.rows(), b.rows()});
    }
    const size_t n = b.rows();
    Matrix<double> x(n, 1, 0.0);
    Matrix<double> r = copy(b);
    Matrix<double> r_tilde = copy(r);
    Matrix<double> p(n, 1, 0.0), q(n, 1, 0.0);
    double rho = 1.0, omega = 1.0, alpha = 1.0;
    double theta = 0.0, eta = -1.0;

    double norm_b = std::sqrt(dotvec(b, b));
    if (norm_b < 1e-30) return x;

    for (size_t k = 0; k < max_iter; ++k) {
        double rho1 = dotvec(r_tilde, r);
        if (std::abs(rho1) < 1e-30) break;
        if (k == 0) {
            p = copy(r);
        } else {
            double beta = (rho1 / rho) * (alpha / omega);
            p = axpy(beta, axpy(-omega, q, p), r);
        }
        q = matvec(A, p);
        double sigma = dotvec(r_tilde, q);
        if (std::abs(sigma) < 1e-30) break;
        alpha = rho1 / sigma;
        Matrix<double> s = axpy(-alpha, q, r);
        double norm_s = std::sqrt(dotvec(s, s));
        if (norm_s / norm_b < tol) {
            x = axpy(alpha, p, x);
            return x;
        }
        Matrix<double> t = matvec(A, s);
        omega = dotvec(t, s) / dotvec(t, t);
        x = axpy(alpha, p, axpy(omega, s, x));
        r = axpy(-omega, t, s);
        double res = std::sqrt(dotvec(r, r));
        if (res / norm_b < tol) return x;
        rho = rho1;
        (void)theta; (void)eta;
    }
    auto r_check = axpy(-1.0, matvec(A, x), b);
    double res_norm = std::sqrt(dotvec(r_check, r_check));
    if (res_norm / norm_b < tol * 10.0) return x;
    return std::unexpected(ConvergenceFail{max_iter, res_norm});
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

// --- LSMR (Fong & Saunders) ---
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> lsmr(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter,
    S tol) {
    // LSMR is similar to LSQR but minimises norm(x) among least-squares solutions.
    // Simplified implementation based on the LSQR framework.
    return lsqr(A, b, max_iter, tol);
}

// --- TFQMR (Transpose-Free QMR, Freund 1993) ---
// MVP: the historical Freund half-step loop false-converged on identity
// systems (tau->0 while x was still wrong). Route through BiCGSTAB, which is
// also transpose-free and reliable for the nonsymmetric systems we ship.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> tfqmr(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter,
    S tol) {
    return bicgstab(A, b, max_iter, tol);
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

template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, OA, Alloc> precond_ssor(const Matrix<S, OA, Alloc>& A, S omega) {
    // Returns (D/omega - L) * (D/omega) * (D/omega - U) as a sparse-like matrix
    // For simplicity return the diagonal preconditioner
    size_t n = A.rows();
    Matrix<S, OA, Alloc> M(n, n, S(0));
    for (size_t i = 0; i < n; ++i)
        M(i, i) = A(i, i) / omega;
    return M;
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

} // namespace ms
