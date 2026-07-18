#include "ms/distributed/iterative.hpp"
#include "ms/distributed/block.hpp"
#include "ms/linalg/linalg.hpp"

#if defined(MS_HAS_MPI) && MS_HAS_MPI
#include <mpi.h>
#endif

namespace ms::distributed {

namespace {

using ColMatrix = Matrix<double, StorageOrder::ColMajor>;

double local_dot(const ColMatrix& a, const ColMatrix& b) {
    double sum = 0.0;
    const size_t n = std::min(a.rows(), b.rows());
    for (size_t i = 0; i < n; ++i) {
        sum += a(i, 0) * b(i, 0);
    }
    return sum;
}

ColMatrix axpy(double alpha, const ColMatrix& x, const ColMatrix& y) {
    ColMatrix r(x.rows(), x.cols());
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            r(i, j) = alpha * x(i, j) + y(i, j);
        }
    }
    return r;
}

ColMatrix copy_matrix(const ColMatrix& m) {
    ColMatrix r(m.rows(), m.cols());
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            r(i, j) = m(i, j);
        }
    }
    return r;
}

template<typename S, template<typename> class Alloc>
Result<ColMatrix> stub_gather_cg(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter,
    S tol) {
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
    return ms::cg(*global_A, *global_b, max_iter, tol);
}

template<typename S, template<typename> class Alloc>
Result<ColMatrix> stub_gather_gmres(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t restart,
    size_t max_iter,
    S tol) {
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
    return ms::gmres(*global_A, *global_b, restart, max_iter, tol);
}

template<typename S, template<typename> class Alloc>
Result<ColMatrix> stub_gather_jacobi(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter,
    S tol) {
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
    return ms::jacobi(*global_A, *global_b, max_iter, tol);
}

template<typename S, template<typename> class Alloc>
Result<ColMatrix> stub_gather_bicgstab(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter,
    S tol) {
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
    return ms::bicgstab(*global_A, *global_b, max_iter, tol);
}

template<typename S, template<typename> class Alloc>
Result<ColMatrix> stub_gather_minres(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter,
    S tol) {
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
    return ms::minres(*global_A, *global_b, max_iter, tol);
}

template<typename S, template<typename> class Alloc>
Result<ColMatrix> stub_gather_qmr(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter,
    S tol) {
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
    return ms::qmr(*global_A, *global_b, max_iter, tol);
}

template<typename S, template<typename> class Alloc>
Result<ColMatrix> stub_gather_tfqmr(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter,
    S tol) {
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
    return ms::tfqmr(*global_A, *global_b, max_iter, tol);
}

template<typename S, template<typename> class Alloc>
Result<ColMatrix> stub_gather_lsmr(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter,
    S tol) {
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
    return ms::lsmr(*global_A, *global_b, max_iter, tol);
}

#if defined(MS_HAS_MPI) && MS_HAS_MPI
Result<ColMatrix> broadcast_vector(const ColMatrix& root, MPIContext& ctx) {
    int rows = 0;
    if (rank(ctx) == 0) {
        rows = static_cast<int>(root.rows());
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

    ColMatrix out(static_cast<size_t>(rows), 1);
    if (rank(ctx) == 0) {
        out = root;
    }
    if (rows > 0) {
        MPI_Bcast(out.data(), rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    return out;
}

template<typename S, template<typename> class Alloc>
Result<ColMatrix> replicate_vector(const DistMatrix<S, Alloc>& dist, MPIContext& ctx) {
    ColMatrix root;
    if (rank(ctx) == 0) {
        auto global = gather(dist, ctx);
        if (!global) {
            return std::unexpected(global.error());
        }
        root = std::move(*global);
    }
    return broadcast_vector(root, ctx);
}

template<typename S, template<typename> class Alloc>
Result<ColMatrix> row_block_matvec(
    const DistMatrix<S, Alloc>& A,
    const ColMatrix& x,
    MPIContext& ctx) {
    if (A.global_cols != x.rows()) {
        return std::unexpected(DimensionMismatch{A.global_cols, x.rows()});
    }
    auto local_y = ms::matmul(A.local, x);
    if (!local_y) {
        return std::unexpected(local_y.error());
    }
    return *local_y;
}

double dot_replicated_with_local(
    const ColMatrix& replicated,
    const ColMatrix& local,
    const DistMatrix<double>& layout,
    MPIContext& ctx) {
    double local_sum = 0.0;
    if (layout.distribution == Distribution::BlockCyclic) {
        for (size_t i = 0; i < local.rows(); ++i) {
            const size_t gi = layout.row_map[i];
            local_sum += replicated(gi, 0) * local(i, 0);
        }
    } else {
        const RowExtent ext = block_row_extent(layout.global_rows, rank(ctx), size(ctx));
        for (size_t i = 0; i < local.rows(); ++i) {
            local_sum += replicated(ext.start + i, 0) * local(i, 0);
        }
    }
    return allreduce_sum(ctx, local_sum);
}

template<typename S, template<typename> class Alloc>
Result<ColMatrix> distributed_matvec_full(
    const DistMatrix<S, Alloc>& A,
    const ColMatrix& x,
    MPIContext& ctx) {
    auto local_y = row_block_matvec(A, x, ctx);
    if (!local_y) {
        return std::unexpected(local_y.error());
    }
    return replicate_vector(
        DistMatrix<double>{
            *local_y,
            A.distribution,
            rank(ctx),
            A.global_rows,
            1,
            A.row_map},
        ctx);
}

template<typename S, template<typename> class Alloc>
Result<ColMatrix> mpi_block_jacobi(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter,
    S tol) {
    if (A.distribution != Distribution::Block) {
        return stub_gather_jacobi(A, b, ctx, max_iter, tol);
    }

    const size_t n = A.global_rows;
    if (n != A.global_cols || n != b.global_rows || b.global_cols != 1) {
        return std::unexpected(DimensionMismatch{A.global_rows, b.global_rows});
    }

    ColMatrix x(n, 1, 0.0);
    double rsold = allreduce_sum(ctx, local_dot(b.local, b.local));

    for (size_t k = 0; k < max_iter; ++k) {
        auto Ax = row_block_matvec(A, x, ctx);
        if (!Ax) {
            return std::unexpected(Ax.error());
        }

        const RowExtent ext = block_row_extent(A.global_rows, rank(ctx), size(ctx));
        for (size_t li = 0; li < A.local.rows(); ++li) {
            const size_t gi = ext.start + li;
            if (std::abs(A.local(li, gi)) < 1e-30) {
                return std::unexpected(DomainError{"dist_jacobi", "zero diagonal entry"});
            }
            x(gi, 0) = (b.local(li, 0) - (*Ax)(li, 0) + A.local(li, gi) * x(gi, 0)) /
                       A.local(li, gi);
        }

        ColMatrix x_local(A.local.rows(), 1);
        for (size_t li = 0; li < A.local.rows(); ++li) {
            x_local(li, 0) = x(ext.start + li, 0);
        }
        auto x_full = replicate_vector(
            DistMatrix<double>{x_local, A.distribution, rank(ctx), n, 1, A.row_map},
            ctx);
        if (!x_full) {
            return std::unexpected(x_full.error());
        }
        x = std::move(*x_full);

        Ax = row_block_matvec(A, x, ctx);
        if (!Ax) {
            return std::unexpected(Ax.error());
        }
        ColMatrix r_local = copy_matrix(b.local);
        for (size_t i = 0; i < r_local.rows(); ++i) {
            r_local(i, 0) -= (*Ax)(i, 0);
        }
        const double rsnew = allreduce_sum(ctx, local_dot(r_local, r_local));
        if (std::sqrt(rsnew) < tol) {
            return x;
        }
        rsold = rsnew;
    }

    return std::unexpected(ConvergenceFail{max_iter, std::sqrt(rsold)});
}

template<typename S, template<typename> class Alloc>
Result<ColMatrix> mpi_block_gmres(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t restart,
    size_t max_iter,
    S tol) {
    if (A.distribution != Distribution::Block) {
        return stub_gather_gmres(A, b, ctx, restart, max_iter, tol);
    }

    const size_t n = A.global_rows;
    if (n != A.global_cols || n != b.global_rows || b.global_cols != 1) {
        return std::unexpected(DimensionMismatch{A.global_rows, b.global_rows});
    }

    ColMatrix x(n, 1, 0.0);
    ColMatrix r(n, 1, 0.0);
    {
        auto Ax = row_block_matvec(A, x, ctx);
        if (!Ax) {
            return std::unexpected(Ax.error());
        }
        ColMatrix r_local = copy_matrix(b.local);
        for (size_t i = 0; i < r_local.rows(); ++i) {
            r_local(i, 0) -= (*Ax)(i, 0);
        }
        auto r_rep = replicate_vector(
            DistMatrix<double>{r_local, A.distribution, rank(ctx), n, 1, A.row_map},
            ctx);
        if (!r_rep) {
            return std::unexpected(r_rep.error());
        }
        r = std::move(*r_rep);
    }

    for (size_t outer = 0; outer < max_iter; outer += restart) {
        const double beta = std::sqrt(local_dot(r, r));
        if (beta < tol) {
            return x;
        }

        std::vector<ColMatrix> V(restart + 1);
        V[0] = ColMatrix(n, 1);
        for (size_t i = 0; i < n; ++i) {
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
            auto w_full = distributed_matvec_full(A, V[step], ctx);
            if (!w_full) {
                return std::unexpected(w_full.error());
            }
            ColMatrix w = std::move(*w_full);
            for (size_t i = 0; i <= step; ++i) {
                H[i][step] = local_dot(V[i], w);
                w = axpy(-H[i][step], V[i], w);
            }
            H[step + 1][step] = std::sqrt(local_dot(w, w));
            const bool invariant = (H[step + 1][step] < 1e-14);
            if (!invariant && step + 1 < restart) {
                V[step + 1] = ColMatrix(n, 1);
                for (size_t i = 0; i < n; ++i) {
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
            double sum = g[i];
            for (size_t k = i + 1; k < j; ++k) {
                sum -= H[i][k] * y(k, 0);
            }
            y(i, 0) = sum / H[i][i];
        }

        for (size_t i = 0; i < j; ++i) {
            x = axpy(y(i, 0), V[i], x);
        }

        {
            auto Ax = row_block_matvec(A, x, ctx);
            if (!Ax) {
                return std::unexpected(Ax.error());
            }
            ColMatrix r_local = copy_matrix(b.local);
            for (size_t i = 0; i < r_local.rows(); ++i) {
                r_local(i, 0) -= (*Ax)(i, 0);
            }
            auto r_rep = replicate_vector(
                DistMatrix<double>{r_local, A.distribution, rank(ctx), n, 1, A.row_map},
                ctx);
            if (!r_rep) {
                return std::unexpected(r_rep.error());
            }
            r = std::move(*r_rep);
        }
        if (std::sqrt(local_dot(r, r)) < tol) {
            return x;
        }
    }

    return std::unexpected(ConvergenceFail{max_iter, std::sqrt(local_dot(r, r))});
}

template<typename S, template<typename> class Alloc>
Result<ColMatrix> mpi_block_cg(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx,
    size_t max_iter,
    S tol) {
    if (A.distribution != Distribution::Block) {
        return stub_gather_cg(A, b, ctx, max_iter, tol);
    }

    const size_t n = A.global_rows;
    if (n != A.global_cols || n != b.global_rows || b.global_cols != 1) {
        return std::unexpected(DimensionMismatch{A.global_rows, b.global_rows});
    }

    ColMatrix x(n, 1, 0.0);
    auto Ap = row_block_matvec(A, x, ctx);
    if (!Ap) {
        return std::unexpected(Ap.error());
    }
    ColMatrix r_local = copy_matrix(b.local);
    for (size_t i = 0; i < r_local.rows(); ++i) {
        r_local(i, 0) -= (*Ap)(i, 0);
    }

    auto r_full = replicate_vector(
        DistMatrix<double>{
            r_local,
            A.distribution,
            rank(ctx),
            n,
            1,
            A.row_map},
        ctx);
    if (!r_full) {
        return std::unexpected(r_full.error());
    }

    ColMatrix p = copy_matrix(*r_full);
    double rsold = allreduce_sum(ctx, local_dot(r_local, r_local));

    for (size_t k = 0; k < max_iter; ++k) {
        Ap = row_block_matvec(A, p, ctx);
        if (!Ap) {
            return std::unexpected(Ap.error());
        }
        const double pAp = dot_replicated_with_local(p, *Ap, A, ctx);
        if (std::abs(pAp) < 1e-30) {
            return std::unexpected(DomainError{"dist_cg", "matrix not symmetric or indefinite"});
        }
        const double alpha = rsold / pAp;
        x = axpy(alpha, p, x);
        r_local = axpy(-alpha, *Ap, r_local);

        const double rsnew = allreduce_sum(ctx, local_dot(r_local, r_local));
        if (std::sqrt(rsnew) < tol) {
            return x;
        }

        r_full = replicate_vector(
            DistMatrix<double>{
                r_local,
                A.distribution,
                rank(ctx),
                n,
                1,
                A.row_map},
            ctx);
        if (!r_full) {
            return std::unexpected(r_full.error());
        }
        p = axpy(rsnew / rsold, p, *r_full);
        rsold = rsnew;
    }

    return std::unexpected(ConvergenceFail{max_iter, std::sqrt(rsold)});
}
#endif

} // namespace

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

#if defined(MS_HAS_MPI) && MS_HAS_MPI
    if (size(ctx) > 1) {
        return mpi_block_cg(A, b, ctx, max_iter, tol);
    }
#endif

    return stub_gather_cg(A, b, ctx, max_iter, tol);
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

#if defined(MS_HAS_MPI) && MS_HAS_MPI
    if (size(ctx) > 1) {
        return mpi_block_gmres(A, b, ctx, restart, max_iter, tol);
    }
#endif

    return stub_gather_gmres(A, b, ctx, restart, max_iter, tol);
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

#if defined(MS_HAS_MPI) && MS_HAS_MPI
    if (size(ctx) > 1) {
        return mpi_block_jacobi(A, b, ctx, max_iter, tol);
    }
#endif

    return stub_gather_jacobi(A, b, ctx, max_iter, tol);
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

    return stub_gather_bicgstab(A, b, ctx, max_iter, tol);
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

    return stub_gather_minres(A, b, ctx, max_iter, tol);
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

    return stub_gather_qmr(A, b, ctx, max_iter, tol);
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

    return stub_gather_tfqmr(A, b, ctx, max_iter, tol);
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

    return stub_gather_lsmr(A, b, ctx, max_iter, tol);
}

template Result<Matrix<double>> dist_lsmr(
    const DistMatrix<double>&,
    const DistMatrix<double>&,
    MPIContext&,
    size_t,
    double);

} // namespace ms::distributed
