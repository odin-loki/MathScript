#include "ms/distributed/iterative.hpp"
#include "ms/distributed/block.hpp"
#include "ms/distributed/dist_ops.hpp"
#include "ms/linalg/linalg.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace ms::distributed {

namespace {

using ColMatrix = Matrix<double, StorageOrder::ColMajor>;

// Magnitude below which a scalar counts as an exact zero (breakdown). The same
// constant the serial kernels in src/linalg/iterative.cpp use, so the
// distributed loops break in exactly the same places.
constexpr double kTiny = 1e-30;

// The distributed loops are double-only because MPI_DOUBLE is the wire type, so
// the templated entry points convert element-wise. For the only instantiation
// (S = double) this is a plain O(m*n) copy.
template<typename S, template<typename> class Alloc>
ColMatrix to_double_block(const Matrix<S, StorageOrder::ColMajor, Alloc>& m) {
    ColMatrix out(m.rows(), m.cols());
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            out(i, j) = static_cast<double>(m(i, j));
        }
    }
    return out;
}

template<typename S, template<typename> class Alloc>
Matrix<S, StorageOrder::ColMajor, Alloc> from_double_block(const ColMatrix& m) {
    Matrix<S, StorageOrder::ColMajor, Alloc> out(m.rows(), m.cols());
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            out(i, j) = static_cast<S>(m(i, j));
        }
    }
    return out;
}

// True when the row-distributed loops can run on these operands: contiguous
// block rows that match this rank's share of the layout, with every column of A
// present locally. BlockCyclic and hand-built DistMatrix objects (dimensions
// set, `local` empty) deliberately fail this and take the gather fallback.
template<typename S, template<typename> class Alloc>
bool rowblock_applicable(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    const MPIContext& ctx) {
    if (size(ctx) > 1 && !mpi_active(ctx)) {
        return false;
    }
    if (A.distribution != Distribution::Block ||
        b.distribution != Distribution::Block) {
        return false;
    }
    const RowLayout lay = make_row_layout(A.global_rows, size(ctx));
    if (!layout_matches(lay, rank(ctx), A.local.rows()) ||
        !layout_matches(lay, rank(ctx), b.local.rows())) {
        return false;
    }
    if (A.local.cols() != A.global_cols) {
        return false;
    }
    if (A.global_rows != 0 && b.local.cols() != 1) {
        return false;
    }
    return true;
}

// The fallback every solver shares: gather both operands and run the serial
// kernel on rank 0. Byte for byte what this module did before the row-block
// loops existed, including handing back the gathered right-hand side on the
// other ranks.
template<typename S, template<typename> class Alloc, typename Fn>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> stub_gather_solve(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    const Fn& kernel) {
    auto global_A = gather(A, ctx);
    if (!global_A) {
        return std::unexpected(global_A.error());
    }
    auto global_b = gather(b, ctx);
    if (!global_b) {
        return std::unexpected(global_b.error());
    }
    if (rank(ctx) != 0 && size(ctx) > 1) {
        return *global_b;
    }
    return kernel(*global_A, *global_b);
}

// b - A*x for the row-distributed operands.
Result<DistVec> dist_residual(
    const MPIContext& ctx,
    const ColMatrix& A,
    const DistVec& x,
    const DistVec& b,
    const RowLayout& lay) {
    auto Ax = dist_matvec(ctx, A, x, lay);
    if (!Ax) {
        return std::unexpected(Ax.error());
    }
    return dist_axpy(-1.0, *Ax, b);
}

// ---------------------------------------------------------------------------
// Conjugate gradient — mirrors ms::cg operation for operation.
// ---------------------------------------------------------------------------

Result<DistVec> dist_cg_impl(
    const MPIContext& ctx,
    const ColMatrix& A,
    const DistVec& b,
    const RowLayout& lay,
    size_t max_iter,
    double tol) {
    if (lay.global_rows == 0) {
        return DistVec(0, 1);
    }
    if (!dist_is_symmetric(ctx, A, lay, 1e-10)) {
        return std::unexpected(DomainError{"dist_cg", "matrix not symmetric"});
    }

    DistVec x(b.rows(), 1, 0.0);
    DistVec r = dist_copy(b);
    DistVec p = dist_copy(r);
    double rsold = dist_dot(ctx, r, r);

    for (size_t k = 0; k < max_iter; ++k) {
        auto Ap = dist_matvec(ctx, A, p, lay);
        if (!Ap) {
            return std::unexpected(Ap.error());
        }
        // ms::cg divides by p'Ap unguarded; a guard here would change the error
        // type on degenerate input, so parity means reproducing its absence.
        const double alpha = rsold / dist_dot(ctx, p, *Ap);
        x = dist_axpy(alpha, p, x);
        r = dist_axpy(-alpha, *Ap, r);
        const double rsnew = dist_dot(ctx, r, r);
        if (std::sqrt(rsnew) < tol) {
            return x;
        }
        p = dist_axpy(rsnew / rsold, p, r);
        rsold = rsnew;
    }

    return std::unexpected(ConvergenceFail{max_iter, std::sqrt(rsold)});
}

// ---------------------------------------------------------------------------
// GMRES(m) — mirrors ms::gmres. Only the Arnoldi basis is distributed; the
// Hessenberg matrix, the Givens rotations and the back substitution are
// redundant on every rank, driven by allreduce results the MPI standard
// requires to be identical everywhere.
// ---------------------------------------------------------------------------

Result<DistVec> dist_gmres_impl(
    const MPIContext& ctx,
    const ColMatrix& A,
    const DistVec& b,
    const RowLayout& lay,
    size_t restart,
    size_t max_iter,
    double tol) {
    if (lay.global_rows == 0) {
        return DistVec(0, 1);
    }

    const size_t m_loc = b.rows();
    DistVec x(m_loc, 1, 0.0);
    DistVec r = dist_copy(b);

    for (size_t outer = 0; outer < max_iter; outer += restart) {
        const double beta = dist_norm(ctx, r);
        if (beta < tol) {
            return x;
        }

        std::vector<DistVec> V(restart + 1);
        V[0] = DistVec(m_loc, 1);
        for (size_t i = 0; i < m_loc; ++i) {
            V[0](i, 0) = r(i, 0) / beta;
        }

        std::vector<double> g(restart + 1, 0.0);
        g[0] = beta;
        std::vector<std::vector<double>> H(
            restart + 1, std::vector<double>(restart, 0.0));
        std::vector<double> cs(restart, 0.0);
        std::vector<double> sn(restart, 0.0);

        size_t j = 0;
        const size_t iter_limit = std::min(restart, max_iter - outer);
        for (size_t step = 0; step < iter_limit; ++step) {
            auto w_res = dist_matvec(ctx, A, V[step], lay);
            if (!w_res) {
                return std::unexpected(w_res.error());
            }
            DistVec w = std::move(*w_res);
            for (size_t i = 0; i <= step; ++i) {
                H[i][step] = dist_dot(ctx, V[i], w);
                w = dist_axpy(-H[i][step], V[i], w);
            }
            H[step + 1][step] = dist_norm(ctx, w);
            const bool invariant = (H[step + 1][step] < 1e-14);
            // ms::gmres never builds V[restart]; reproducing that quirk is what
            // keeps the answers identical.
            if (!invariant && step + 1 < restart) {
                V[step + 1] = DistVec(m_loc, 1);
                for (size_t i = 0; i < m_loc; ++i) {
                    V[step + 1](i, 0) = w(i, 0) / H[step + 1][step];
                }
            }

            for (size_t i = 0; i < step; ++i) {
                const double h0 = cs[i] * H[i][step] + sn[i] * H[i + 1][step];
                const double h1 = -sn[i] * H[i][step] + cs[i] * H[i + 1][step];
                H[i][step] = h0;
                H[i + 1][step] = h1;
            }
            const double denom = std::sqrt(H[step][step] * H[step][step] +
                                           H[step + 1][step] * H[step + 1][step]);
            if (denom >= 1e-14) {
                cs[step] = H[step][step] / denom;
                sn[step] = H[step + 1][step] / denom;
                H[step][step] = denom;
                H[step + 1][step] = 0.0;
                const double g0 = g[step];
                g[step] = cs[step] * g0;
                g[step + 1] = -sn[step] * g0;
            }

            j = step + 1;
            if (invariant || std::abs(g[step + 1]) < tol) {
                break;
            }
        }

        ColMatrix y(j, 1, 0.0);
        for (int i = static_cast<int>(j) - 1; i >= 0; --i) {
            const size_t ui = static_cast<size_t>(i);
            double sum = g[ui];
            for (size_t k = ui + 1; k < j; ++k) {
                sum -= H[ui][k] * y(k, 0);
            }
            y(ui, 0) = sum / H[ui][ui];
        }
        for (size_t i = 0; i < j; ++i) {
            x = dist_axpy(y(i, 0), V[i], x);
        }

        auto r_next = dist_residual(ctx, A, x, b, lay);
        if (!r_next) {
            return std::unexpected(r_next.error());
        }
        r = std::move(*r_next);
        if (dist_norm(ctx, r) < tol) {
            return x;
        }
    }

    return std::unexpected(ConvergenceFail{max_iter, dist_norm(ctx, r)});
}

// ---------------------------------------------------------------------------
// Jacobi — mirrors ms::jacobi. The diagonal entry of local row i is
// A(i, starts[my_rank] + i), always present because the row block carries all
// n columns, so the sweep needs no extra communication of its own.
// ---------------------------------------------------------------------------

Result<DistVec> dist_jacobi_impl(
    const MPIContext& ctx,
    const ColMatrix& A,
    const DistVec& b,
    const RowLayout& lay,
    size_t max_iter,
    double tol) {
    if (lay.global_rows == 0) {
        return DistVec(0, 1);
    }
    if (rank(ctx) < 0 || rank(ctx) >= lay.nprocs) {
        return std::unexpected(
            DomainError{"dist_jacobi", "rank outside the row layout"});
    }

    const size_t row0 = lay.starts[static_cast<size_t>(rank(ctx))];
    DistVec x(b.rows(), 1, 0.0);
    double res_norm = dist_norm(ctx, b);

    for (size_t k = 0; k < max_iter; ++k) {
        auto Ax = dist_matvec(ctx, A, x, lay);
        if (!Ax) {
            return std::unexpected(Ax.error());
        }

        DistVec x_new(b.rows(), 1, 0.0);
        bool local_ok = true;
        for (size_t i = 0; i < b.rows(); ++i) {
            const double d = A(i, row0 + i);
            if (std::abs(d) < kTiny) {
                local_ok = false;
                break;
            }
            x_new(i, 0) = (b(i, 0) - (*Ax)(i, 0) + d * x(i, 0)) / d;
        }
        // Every rank has to reach the same verdict, or the ranks that did find
        // a usable diagonal would go on to a collective nobody else enters.
        if (!dist_all_true(ctx, local_ok)) {
            return std::unexpected(
                DomainError{"dist_jacobi", "zero diagonal entry"});
        }

        auto r = dist_residual(ctx, A, x_new, b, lay);
        if (!r) {
            return std::unexpected(r.error());
        }
        res_norm = dist_norm(ctx, *r);
        if (res_norm < tol) {
            return x_new;
        }
        x = std::move(x_new);
    }

    return std::unexpected(ConvergenceFail{max_iter, res_norm});
}

// ---------------------------------------------------------------------------
// BiCGSTAB — mirrors ms::bicgstab, which always returns its current iterate and
// never reports ConvergenceFail.
// ---------------------------------------------------------------------------

Result<DistVec> dist_bicgstab_impl(
    const MPIContext& ctx,
    const ColMatrix& A,
    const DistVec& b,
    const RowLayout& lay,
    size_t max_iter,
    double tol) {
    if (lay.global_rows == 0) {
        return DistVec(0, 1);
    }

    DistVec x(b.rows(), 1, 0.0);
    DistVec r = dist_copy(b);
    DistVec r0 = dist_copy(r);
    DistVec p = dist_copy(r);
    DistVec v(b.rows(), 1, 0.0);

    double rho = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    for (size_t k = 0; k < max_iter; ++k) {
        const double rho1 = dist_dot(ctx, r0, r);
        if (std::abs(rho1) < kTiny) {
            break;
        }
        if (k == 0) {
            p = dist_copy(r);
        } else {
            const double beta = (rho1 / rho) * (alpha / omega);
            const DistVec ph = dist_axpy(-omega, v, p);
            p = dist_axpy(beta, ph, r);
        }

        auto v_res = dist_matvec(ctx, A, p, lay);
        if (!v_res) {
            return std::unexpected(v_res.error());
        }
        v = std::move(*v_res);
        alpha = rho1 / dist_dot(ctx, r0, v);

        const DistVec s = dist_axpy(-alpha, v, r);
        if (dist_norm(ctx, s) < tol) {
            x = dist_axpy(alpha, p, x);
            break;
        }

        auto t_res = dist_matvec(ctx, A, s, lay);
        if (!t_res) {
            return std::unexpected(t_res.error());
        }
        const DistVec t = std::move(*t_res);
        omega = dist_dot(ctx, t, s) / dist_dot(ctx, t, t);
        x = dist_axpy(alpha, p, dist_axpy(omega, s, x));
        r = dist_axpy(-omega, t, s);
        if (dist_norm(ctx, r) < tol) {
            break;
        }
        rho = rho1;
    }

    return x;
}

// ---------------------------------------------------------------------------
// MINRES — mirrors ms::minres exactly, including its |eta| stopping estimate,
// which declares convergence early on most systems. Fixing that belongs in
// src/linalg/iterative.cpp; here parity is the contract.
// ---------------------------------------------------------------------------

Result<DistVec> dist_minres_impl(
    const MPIContext& ctx,
    const ColMatrix& A,
    const DistVec& b,
    const RowLayout& lay,
    size_t max_iter,
    double tol) {
    if (lay.global_rows == 0) {
        return DistVec(0, 1);
    }

    const size_t m_loc = b.rows();
    DistVec x(m_loc, 1, 0.0);
    DistVec v = dist_copy(b);
    double beta = dist_norm(ctx, v);
    if (beta < 1e-14) {
        return x;
    }
    for (size_t i = 0; i < m_loc; ++i) {
        v(i, 0) /= beta;
    }

    DistVec v_prev(m_loc, 1, 0.0);
    double c_old = 1.0;
    double c_cur = 1.0;
    double s_old = 0.0;
    double s_cur = 0.0;
    double eta = beta;
    DistVec w(m_loc, 1, 0.0);
    DistVec w_prev(m_loc, 1, 0.0);

    for (size_t iter = 0; iter < max_iter; ++iter) {
        auto Av_res = dist_matvec(ctx, A, v, lay);
        if (!Av_res) {
            return std::unexpected(Av_res.error());
        }
        const DistVec Av = std::move(*Av_res);
        const double alpha = dist_dot(ctx, v, Av);

        DistVec v_next = dist_axpy(-alpha, v, dist_axpy(-beta, v_prev, Av));
        const double beta_next = dist_norm(ctx, v_next);
        if (beta_next > 1e-14) {
            for (size_t i = 0; i < m_loc; ++i) {
                v_next(i, 0) /= beta_next;
            }
        }

        const double delta = c_cur * alpha - c_old * s_cur * beta;
        double gamma = std::sqrt(delta * delta + beta_next * beta_next);
        if (gamma < 1e-14) {
            gamma = 1e-14;
        }
        const double c_new = delta / gamma;
        const double s_new = beta_next / gamma;

        const DistVec w_new = dist_axpy(
            -c_old * s_cur * beta / gamma, w_prev,
            dist_axpy(-s_old * beta / gamma, w, dist_copy(v)));
        x = dist_axpy(c_new * eta, w_new, x);
        eta = -s_new * eta;

        w_prev = w;
        w = w_new;
        v_prev = v;
        v = v_next;
        beta = beta_next;
        c_old = c_cur;
        s_old = s_cur;
        c_cur = c_new;
        s_cur = s_new;

        if (std::abs(eta) < tol) {
            return x;
        }
    }

    auto r_check = dist_residual(ctx, A, x, b, lay);
    if (!r_check) {
        return std::unexpected(r_check.error());
    }
    const double res_norm = dist_norm(ctx, *r_check);
    if (res_norm < tol * 10.0) {
        return x;
    }
    return std::unexpected(ConvergenceFail{max_iter, res_norm});
}

// ---------------------------------------------------------------------------
// QMR — mirrors ms::qmr, the two-sided Lanczos formulation without look-ahead.
// A^T q is the only place the vector spaces cross: dist_matvec_transpose sums
// the column partials into a replicated length-n vector, and this rank's block
// row of that is what the recurrence needs.
// ---------------------------------------------------------------------------

Result<DistVec> dist_qmr_impl(
    const MPIContext& ctx,
    const ColMatrix& A,
    const DistVec& b,
    const RowLayout& lay,
    size_t max_iter,
    double tol) {
    if (lay.global_rows == 0) {
        return DistVec(0, 1);
    }

    const size_t m_loc = b.rows();
    DistVec x(m_loc, 1, 0.0);

    const double norm_b = dist_norm(ctx, b);
    if (norm_b < kTiny) {
        return x;
    }

    DistVec r = dist_copy(b);
    DistVec v_tilde = dist_copy(r);
    DistVec w_tilde = dist_copy(r);
    double rho = dist_norm(ctx, v_tilde);
    double xi = dist_norm(ctx, w_tilde);

    double gamma = 1.0;
    double eta = -1.0;
    double theta = 0.0;
    double epsilon = 1.0;

    DistVec p(m_loc, 1, 0.0);
    DistVec q(m_loc, 1, 0.0);
    DistVec d(m_loc, 1, 0.0);
    DistVec s(m_loc, 1, 0.0);

    size_t used = 0;
    for (size_t k = 1; k <= max_iter; ++k) {
        used = k;
        if (std::abs(rho) < kTiny || std::abs(xi) < kTiny) {
            break;  // the Lanczos vectors have died out
        }
        const DistVec v = dist_scale(1.0 / rho, v_tilde);
        const DistVec w = dist_scale(1.0 / xi, w_tilde);

        const double delta = dist_dot(ctx, w, v);
        if (std::abs(delta) < kTiny) {
            break;  // serious breakdown; look-ahead would be needed here
        }
        if (k == 1) {
            p = dist_copy(v);
            q = dist_copy(w);
        } else {
            p = dist_axpy(-(xi * delta / epsilon), p, v);
            q = dist_axpy(-(rho * delta / epsilon), q, w);
        }

        auto pt_res = dist_matvec(ctx, A, p, lay);
        if (!pt_res) {
            return std::unexpected(pt_res.error());
        }
        const DistVec p_tilde = std::move(*pt_res);
        epsilon = dist_dot(ctx, q, p_tilde);
        if (std::abs(epsilon) < kTiny) {
            break;
        }
        const double beta = epsilon / delta;
        if (std::abs(beta) < kTiny) {
            break;
        }

        v_tilde = dist_axpy(-beta, v, p_tilde);
        const double rho_next = dist_norm(ctx, v_tilde);

        auto atq = dist_matvec_transpose(ctx, A, q);
        if (!atq) {
            return std::unexpected(atq.error());
        }
        w_tilde = dist_axpy(-beta, w, local_segment(*atq, lay, rank(ctx)));
        const double xi_next = dist_norm(ctx, w_tilde);

        // theta uses the NEW rho against the previous gamma; eta uses the OLD
        // rho, so rho is only shifted once eta has been formed.
        const double theta_prev = theta;
        const double gamma_prev = gamma;
        theta = rho_next / (gamma_prev * std::abs(beta));
        gamma = 1.0 / std::sqrt(1.0 + theta * theta);
        if (std::abs(gamma) < kTiny) {
            break;
        }
        eta = -eta * rho * gamma * gamma / (beta * gamma_prev * gamma_prev);

        if (k == 1) {
            d = dist_scale(eta, p);
            s = dist_scale(eta, p_tilde);
        } else {
            const double c = (theta_prev * gamma) * (theta_prev * gamma);
            d = dist_axpy(c, d, dist_scale(eta, p));
            s = dist_axpy(c, s, dist_scale(eta, p_tilde));
        }
        x = dist_axpy(1.0, d, x);
        r = dist_axpy(-1.0, s, r);  // s == A*d in exact arithmetic

        rho = rho_next;
        xi = xi_next;

        // The recursive r drifts from the true residual, so confirm first.
        if (dist_norm(ctx, r) <= tol * norm_b) {
            auto res = dist_residual(ctx, A, x, b, lay);
            if (!res) {
                return std::unexpected(res.error());
            }
            if (dist_norm(ctx, *res) <= tol * norm_b) {
                return x;
            }
        }
    }

    auto res = dist_residual(ctx, A, x, b, lay);
    if (!res) {
        return std::unexpected(res.error());
    }
    const double res_norm = dist_norm(ctx, *res);
    if (res_norm <= 10.0 * tol * norm_b) {
        return x;
    }
    return std::unexpected(ConvergenceFail{used, res_norm});
}

// ---------------------------------------------------------------------------
// TFQMR — mirrors ms::tfqmr: the CGS recurrence smoothed by QMR's rotations, so
// only products with A are needed. x is updated BEFORE the quasi-residual test
// and the true residual is what declares convergence.
// ---------------------------------------------------------------------------

Result<DistVec> dist_tfqmr_impl(
    const MPIContext& ctx,
    const ColMatrix& A,
    const DistVec& b,
    const RowLayout& lay,
    size_t max_iter,
    double tol) {
    if (lay.global_rows == 0) {
        return DistVec(0, 1);
    }

    const size_t m_loc = b.rows();
    DistVec x(m_loc, 1, 0.0);

    const double norm_b = dist_norm(ctx, b);
    if (norm_b < kTiny) {
        return x;
    }

    DistVec r = dist_copy(b);
    DistVec w = dist_copy(r);
    DistVec y_odd = dist_copy(r);
    const DistVec r_shadow = dist_copy(r);

    auto u_first = dist_matvec(ctx, A, y_odd, lay);
    if (!u_first) {
        return std::unexpected(u_first.error());
    }
    DistVec u_odd = std::move(*u_first);
    DistVec v = dist_copy(u_odd);
    DistVec d(m_loc, 1, 0.0);
    DistVec y_even(m_loc, 1, 0.0);
    DistVec u_even(m_loc, 1, 0.0);

    double tau = dist_norm(ctx, r);
    double theta = 0.0;
    double eta = 0.0;
    double rho = dist_dot(ctx, r_shadow, r);
    size_t m = 0;  // half-step counter

    for (size_t it = 1; it <= max_iter; ++it) {
        if (std::abs(rho) < kTiny) {
            break;
        }
        const double sigma = dist_dot(ctx, r_shadow, v);
        if (std::abs(sigma) < kTiny) {
            break;
        }
        const double alpha = rho / sigma;
        if (std::abs(alpha) < kTiny || !std::isfinite(alpha)) {
            break;
        }

        y_even = dist_axpy(-alpha, v, y_odd);
        auto ue = dist_matvec(ctx, A, y_even, lay);
        if (!ue) {
            return std::unexpected(ue.error());
        }
        u_even = std::move(*ue);

        bool converged = false;
        bool stalled = false;
        for (size_t half = 0; half < 2; ++half) {
            const DistVec& y_m = (half == 0) ? y_odd : y_even;
            const DistVec& u_m = (half == 0) ? u_odd : u_even;
            ++m;
            w = dist_axpy(-alpha, u_m, w);

            if (tau < kTiny || !std::isfinite(tau)) {
                stalled = true;
                break;
            }
            const double theta_prev = theta;
            const double eta_prev = eta;
            theta = dist_norm(ctx, w) / tau;
            const double c = 1.0 / std::sqrt(1.0 + theta * theta);
            tau = tau * theta * c;
            eta = c * c * alpha;

            const double dcoef = theta_prev * theta_prev * eta_prev / alpha;
            d = dist_axpy(dcoef, d, y_m);
            x = dist_axpy(eta, d, x);

            if (tau <= tol * norm_b) {
                auto res = dist_residual(ctx, A, x, b, lay);
                if (!res) {
                    return std::unexpected(res.error());
                }
                if (dist_norm(ctx, *res) <= tol * norm_b) {
                    converged = true;
                    break;
                }
                if (tau < kTiny) {
                    // The quasi-residual collapsed with the true residual still
                    // too large: this recurrence has no progress left.
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

        const double rho_next = dist_dot(ctx, r_shadow, w);
        const double beta = rho_next / rho;
        rho = rho_next;
        y_odd = dist_axpy(beta, y_even, w);
        auto uo = dist_matvec(ctx, A, y_odd, lay);
        if (!uo) {
            return std::unexpected(uo.error());
        }
        u_odd = std::move(*uo);
        v = dist_axpy(beta, dist_axpy(beta, v, u_even), u_odd);
    }

    auto res = dist_residual(ctx, A, x, b, lay);
    if (!res) {
        return std::unexpected(res.error());
    }
    const double res_norm = dist_norm(ctx, *res);
    if (res_norm <= 10.0 * tol * norm_b) {
        return x;
    }
    return std::unexpected(ConvergenceFail{m, res_norm});
}

// ---------------------------------------------------------------------------
// LSQR — mirrors ms::lsqr. The two vector spaces are split: u (length m) is
// block-row distributed, while v, w and x (length n) are REPLICATED, because
// dist_matvec_transpose already produces them replicated via an allreduce.
// So A*v needs no communication at all and only the norms of u and the
// transpose products cross the network.
// ---------------------------------------------------------------------------

Result<DistVec> dist_lsqr_impl(
    const MPIContext& ctx,
    const ColMatrix& A,
    const DistVec& b,
    const RowLayout& lay,
    size_t max_iter,
    double tol) {
    const size_t m_loc = A.rows();
    const size_t n = A.cols();
    if (lay.global_rows == 0 || n == 0) {
        return DistVec(n, 1, 0.0);
    }

    DistVec x(n, 1, 0.0);
    DistVec u = dist_copy(b);
    double beta = dist_norm(ctx, u);
    if (beta < kTiny) {
        return x;
    }
    for (size_t i = 0; i < m_loc; ++i) {
        u(i, 0) /= beta;
    }

    auto v_res = dist_matvec_transpose(ctx, A, u);
    if (!v_res) {
        return std::unexpected(v_res.error());
    }
    DistVec v = std::move(*v_res);
    double alpha = std::sqrt(local_dot(v, v));  // v is replicated: no reduce
    if (alpha < kTiny) {
        return x;
    }
    for (size_t j = 0; j < n; ++j) {
        v(j, 0) /= alpha;
    }

    DistVec w = dist_copy(v);
    double phibar = beta;
    double rhobar = alpha;
    const double norm_b = beta;

    for (size_t k = 0; k < max_iter; ++k) {
        const DistVec Au = local_matvec(A, v);
        u = dist_axpy(-alpha, u, Au);
        beta = dist_norm(ctx, u);
        if (beta > kTiny) {
            for (size_t i = 0; i < m_loc; ++i) {
                u(i, 0) /= beta;
            }
        }

        auto at_res = dist_matvec_transpose(ctx, A, u);
        if (!at_res) {
            return std::unexpected(at_res.error());
        }
        v = dist_axpy(-beta, v, *at_res);
        alpha = std::sqrt(local_dot(v, v));
        if (alpha > kTiny) {
            for (size_t j = 0; j < n; ++j) {
                v(j, 0) /= alpha;
            }
        }

        const double rho = std::sqrt(rhobar * rhobar + beta * beta);
        const double c = rhobar / rho;
        const double s = beta / rho;
        const double theta = s * alpha;
        rhobar = -c * alpha;
        const double phi = c * phibar;
        phibar = s * phibar;

        x = dist_axpy(phi / rho, w, x);
        w = dist_axpy(-theta / rho, w, v);

        if (std::abs(phibar) / norm_b < tol) {
            return x;
        }
    }

    return x;  // ms::lsqr never fails once past its initial guards
}

// ---------------------------------------------------------------------------
// LSMR — mirrors ms::lsmr: the same Golub-Kahan bidiagonalisation as LSQR with
// a second rotation sequence applied to R^T, so ||A^T r|| falls monotonically.
// Same split of the vector spaces as LSQR; x, v, h and hbar are replicated.
// ---------------------------------------------------------------------------

Result<DistVec> dist_lsmr_impl(
    const MPIContext& ctx,
    const ColMatrix& A,
    const DistVec& b,
    const RowLayout& lay,
    size_t max_iter,
    double tol) {
    const size_t n = A.cols();
    if (lay.global_rows == 0 || n == 0) {
        return DistVec(n, 1, 0.0);
    }

    DistVec x(n, 1, 0.0);
    double beta = dist_norm(ctx, b);
    if (beta < kTiny) {
        return x;
    }
    const double norm_b = beta;
    DistVec u = dist_scale(1.0 / beta, b);

    auto v_res = dist_matvec_transpose(ctx, A, u);
    if (!v_res) {
        return std::unexpected(v_res.error());
    }
    DistVec v = std::move(*v_res);
    double alpha = std::sqrt(local_dot(v, v));
    if (alpha < kTiny) {
        return x;  // b is orthogonal to range(A): x = 0 already solves it
    }
    v = dist_scale(1.0 / alpha, v);

    double alphabar = alpha;
    double zetabar = alpha * beta;  // |zetabar| tracks ||A^T r||
    const double norm_ar0 = std::abs(zetabar);
    double rho = 1.0;
    double rhobar = 1.0;
    double cbar = 1.0;
    double sbar = 0.0;
    double zeta = 0.0;

    DistVec h = dist_copy(v);
    DistVec hbar(n, 1, 0.0);

    // State for the ||r|| estimate (Fong & Saunders section 3.3).
    double betadd = beta;
    double betad = 0.0;
    double rhodold = 1.0;
    double tautildeold = 0.0;
    double thetatilde = 0.0;
    double norm_r = beta;

    for (size_t k = 0; k < max_iter; ++k) {
        u = dist_axpy(-alpha, u, local_matvec(A, v));
        beta = dist_norm(ctx, u);
        if (beta > kTiny) {
            u = dist_scale(1.0 / beta, u);
            auto at_res = dist_matvec_transpose(ctx, A, u);
            if (!at_res) {
                return std::unexpected(at_res.error());
            }
            v = dist_axpy(-beta, v, *at_res);
            alpha = std::sqrt(local_dot(v, v));
            if (alpha > kTiny) {
                v = dist_scale(1.0 / alpha, v);
            } else {
                alpha = 0.0;
            }
        } else {
            beta = 0.0;  // the bidiagonalisation has terminated
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

        hbar = dist_axpy(-(thetabar * rho / (rhoold * rhobarold)), hbar, h);
        x = dist_axpy(zeta / (rho * rhobar), hbar, x);
        h = dist_axpy(-(thetanew / rho), h, v);

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

} // namespace

// ---------------------------------------------------------------------------
// Public entry points. Each validates the GLOBAL dimensions first (a caller may
// hand in a DistMatrix whose `local` is empty), then either runs the row-block
// loop or falls back to gather-and-solve, and finally all-gathers the solution
// so every rank returns the same full-length x.
// ---------------------------------------------------------------------------

template<typename S, template<typename> class Alloc>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_cg(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter,
    S tol) {
    if (A.global_rows != A.global_cols) {
        return std::unexpected(DimensionMismatch{A.global_rows, A.global_cols});
    }
    if (A.global_rows != b.global_rows || b.global_cols != 1) {
        return std::unexpected(DimensionMismatch{A.global_rows, b.global_rows});
    }
    if (!rowblock_applicable(A, b, ctx)) {
        return stub_gather_solve(A, b, ctx, [&](const auto& GA, const auto& Gb) {
            return ms::cg(GA, Gb, max_iter, tol);
        });
    }

    const RowLayout lay = make_row_layout(A.global_rows, size(ctx));
    auto x_loc = dist_cg_impl(
        ctx, to_double_block(A.local), to_double_block(b.local), lay, max_iter,
        static_cast<double>(tol));
    if (!x_loc) {
        return std::unexpected(x_loc.error());
    }
    auto x_full = dist_allgather_rows(ctx, *x_loc, lay);
    if (!x_full) {
        return std::unexpected(x_full.error());
    }
    return from_double_block<S, Alloc>(*x_full);
}

template Result<Matrix<double>> dist_cg(
    const DistMatrix<double>&,
    const DistMatrix<double>&,
    MPIContext&,
    size_t,
    double);

template<typename S, template<typename> class Alloc>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_gmres(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t restart,
    size_t max_iter,
    S tol) {
    if (A.global_rows != A.global_cols) {
        return std::unexpected(DimensionMismatch{A.global_rows, A.global_cols});
    }
    if (A.global_rows != b.global_rows || b.global_cols != 1) {
        return std::unexpected(DimensionMismatch{A.global_rows, b.global_rows});
    }
    if (restart == 0) {
        // `outer += restart` would never advance; reject instead of hanging.
        return std::unexpected(
            DomainError{"dist_gmres", "restart must be positive"});
    }
    if (!rowblock_applicable(A, b, ctx)) {
        return stub_gather_solve(A, b, ctx, [&](const auto& GA, const auto& Gb) {
            return ms::gmres(GA, Gb, restart, max_iter, tol);
        });
    }

    const RowLayout lay = make_row_layout(A.global_rows, size(ctx));
    auto x_loc = dist_gmres_impl(
        ctx, to_double_block(A.local), to_double_block(b.local), lay, restart,
        max_iter, static_cast<double>(tol));
    if (!x_loc) {
        return std::unexpected(x_loc.error());
    }
    auto x_full = dist_allgather_rows(ctx, *x_loc, lay);
    if (!x_full) {
        return std::unexpected(x_full.error());
    }
    return from_double_block<S, Alloc>(*x_full);
}

template Result<Matrix<double>> dist_gmres(
    const DistMatrix<double>&,
    const DistMatrix<double>&,
    MPIContext&,
    size_t,
    size_t,
    double);

template<typename S, template<typename> class Alloc>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_jacobi(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter,
    S tol) {
    if (A.global_rows != A.global_cols) {
        return std::unexpected(DimensionMismatch{A.global_rows, A.global_cols});
    }
    if (A.global_rows != b.global_rows || b.global_cols != 1) {
        return std::unexpected(DimensionMismatch{A.global_rows, b.global_rows});
    }
    if (!rowblock_applicable(A, b, ctx)) {
        return stub_gather_solve(A, b, ctx, [&](const auto& GA, const auto& Gb) {
            return ms::jacobi(GA, Gb, max_iter, tol);
        });
    }

    const RowLayout lay = make_row_layout(A.global_rows, size(ctx));
    auto x_loc = dist_jacobi_impl(
        ctx, to_double_block(A.local), to_double_block(b.local), lay, max_iter,
        static_cast<double>(tol));
    if (!x_loc) {
        return std::unexpected(x_loc.error());
    }
    auto x_full = dist_allgather_rows(ctx, *x_loc, lay);
    if (!x_full) {
        return std::unexpected(x_full.error());
    }
    return from_double_block<S, Alloc>(*x_full);
}

template Result<Matrix<double>> dist_jacobi(
    const DistMatrix<double>&,
    const DistMatrix<double>&,
    MPIContext&,
    size_t,
    double);

template<typename S, template<typename> class Alloc>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_bicgstab(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter,
    S tol) {
    if (A.global_rows != A.global_cols) {
        return std::unexpected(DimensionMismatch{A.global_rows, A.global_cols});
    }
    if (A.global_rows != b.global_rows || b.global_cols != 1) {
        return std::unexpected(DimensionMismatch{A.global_rows, b.global_rows});
    }
    if (!rowblock_applicable(A, b, ctx)) {
        return stub_gather_solve(A, b, ctx, [&](const auto& GA, const auto& Gb) {
            return ms::bicgstab(GA, Gb, max_iter, tol);
        });
    }

    const RowLayout lay = make_row_layout(A.global_rows, size(ctx));
    auto x_loc = dist_bicgstab_impl(
        ctx, to_double_block(A.local), to_double_block(b.local), lay, max_iter,
        static_cast<double>(tol));
    if (!x_loc) {
        return std::unexpected(x_loc.error());
    }
    auto x_full = dist_allgather_rows(ctx, *x_loc, lay);
    if (!x_full) {
        return std::unexpected(x_full.error());
    }
    return from_double_block<S, Alloc>(*x_full);
}

template Result<Matrix<double>> dist_bicgstab(
    const DistMatrix<double>&,
    const DistMatrix<double>&,
    MPIContext&,
    size_t,
    double);

template<typename S, template<typename> class Alloc>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_minres(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter,
    S tol) {
    if (A.global_rows != A.global_cols) {
        return std::unexpected(DimensionMismatch{A.global_rows, A.global_cols});
    }
    if (A.global_rows != b.global_rows || b.global_cols != 1) {
        return std::unexpected(DimensionMismatch{A.global_rows, b.global_rows});
    }
    if (!rowblock_applicable(A, b, ctx)) {
        return stub_gather_solve(A, b, ctx, [&](const auto& GA, const auto& Gb) {
            return ms::minres(GA, Gb, max_iter, tol);
        });
    }

    const RowLayout lay = make_row_layout(A.global_rows, size(ctx));
    auto x_loc = dist_minres_impl(
        ctx, to_double_block(A.local), to_double_block(b.local), lay, max_iter,
        static_cast<double>(tol));
    if (!x_loc) {
        return std::unexpected(x_loc.error());
    }
    auto x_full = dist_allgather_rows(ctx, *x_loc, lay);
    if (!x_full) {
        return std::unexpected(x_full.error());
    }
    return from_double_block<S, Alloc>(*x_full);
}

template Result<Matrix<double>> dist_minres(
    const DistMatrix<double>&,
    const DistMatrix<double>&,
    MPIContext&,
    size_t,
    double);

template<typename S, template<typename> class Alloc>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_qmr(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter,
    S tol) {
    if (A.global_rows != A.global_cols) {
        return std::unexpected(DimensionMismatch{A.global_rows, A.global_cols});
    }
    if (A.global_rows != b.global_rows || b.global_cols != 1) {
        return std::unexpected(DimensionMismatch{A.global_rows, b.global_rows});
    }
    if (!rowblock_applicable(A, b, ctx)) {
        return stub_gather_solve(A, b, ctx, [&](const auto& GA, const auto& Gb) {
            return ms::qmr(GA, Gb, max_iter, tol);
        });
    }

    const RowLayout lay = make_row_layout(A.global_rows, size(ctx));
    auto x_loc = dist_qmr_impl(
        ctx, to_double_block(A.local), to_double_block(b.local), lay, max_iter,
        static_cast<double>(tol));
    if (!x_loc) {
        return std::unexpected(x_loc.error());
    }
    auto x_full = dist_allgather_rows(ctx, *x_loc, lay);
    if (!x_full) {
        return std::unexpected(x_full.error());
    }
    return from_double_block<S, Alloc>(*x_full);
}

template Result<Matrix<double>> dist_qmr(
    const DistMatrix<double>&,
    const DistMatrix<double>&,
    MPIContext&,
    size_t,
    double);

template<typename S, template<typename> class Alloc>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_tfqmr(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter,
    S tol) {
    if (A.global_rows != A.global_cols) {
        return std::unexpected(DimensionMismatch{A.global_rows, A.global_cols});
    }
    if (A.global_rows != b.global_rows || b.global_cols != 1) {
        return std::unexpected(DimensionMismatch{A.global_rows, b.global_rows});
    }
    if (!rowblock_applicable(A, b, ctx)) {
        return stub_gather_solve(A, b, ctx, [&](const auto& GA, const auto& Gb) {
            return ms::tfqmr(GA, Gb, max_iter, tol);
        });
    }

    const RowLayout lay = make_row_layout(A.global_rows, size(ctx));
    auto x_loc = dist_tfqmr_impl(
        ctx, to_double_block(A.local), to_double_block(b.local), lay, max_iter,
        static_cast<double>(tol));
    if (!x_loc) {
        return std::unexpected(x_loc.error());
    }
    auto x_full = dist_allgather_rows(ctx, *x_loc, lay);
    if (!x_full) {
        return std::unexpected(x_full.error());
    }
    return from_double_block<S, Alloc>(*x_full);
}

template Result<Matrix<double>> dist_tfqmr(
    const DistMatrix<double>&,
    const DistMatrix<double>&,
    MPIContext&,
    size_t,
    double);

template<typename S, template<typename> class Alloc>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_lsmr(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter,
    S tol) {
    if (A.global_rows != A.global_cols) {
        return std::unexpected(DimensionMismatch{A.global_rows, A.global_cols});
    }
    if (A.global_rows != b.global_rows || b.global_cols != 1) {
        return std::unexpected(DimensionMismatch{A.global_rows, b.global_rows});
    }
    if (!rowblock_applicable(A, b, ctx)) {
        return stub_gather_solve(A, b, ctx, [&](const auto& GA, const auto& Gb) {
            return ms::lsmr(GA, Gb, max_iter, tol);
        });
    }

    const RowLayout lay = make_row_layout(A.global_rows, size(ctx));
    // x is already replicated on every rank, so there is no gather to do.
    auto x = dist_lsmr_impl(
        ctx, to_double_block(A.local), to_double_block(b.local), lay, max_iter,
        static_cast<double>(tol));
    if (!x) {
        return std::unexpected(x.error());
    }
    return from_double_block<S, Alloc>(*x);
}

template Result<Matrix<double>> dist_lsmr(
    const DistMatrix<double>&,
    const DistMatrix<double>&,
    MPIContext&,
    size_t,
    double);

template<typename S, template<typename> class Alloc>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> dist_lsqr(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter,
    S tol) {
    if (A.global_rows != A.global_cols) {
        return std::unexpected(DimensionMismatch{A.global_rows, A.global_cols});
    }
    if (A.global_rows != b.global_rows || b.global_cols != 1) {
        return std::unexpected(DimensionMismatch{A.global_rows, b.global_rows});
    }
    if (!rowblock_applicable(A, b, ctx)) {
        return stub_gather_solve(A, b, ctx, [&](const auto& GA, const auto& Gb) {
            return ms::lsqr(GA, Gb, max_iter, tol);
        });
    }

    const RowLayout lay = make_row_layout(A.global_rows, size(ctx));
    // x is already replicated on every rank, so there is no gather to do.
    auto x = dist_lsqr_impl(
        ctx, to_double_block(A.local), to_double_block(b.local), lay, max_iter,
        static_cast<double>(tol));
    if (!x) {
        return std::unexpected(x.error());
    }
    return from_double_block<S, Alloc>(*x);
}

template Result<Matrix<double>> dist_lsqr(
    const DistMatrix<double>&,
    const DistMatrix<double>&,
    MPIContext&,
    size_t,
    double);

} // namespace ms::distributed
