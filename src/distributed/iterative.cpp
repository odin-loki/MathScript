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

} // namespace ms::distributed
