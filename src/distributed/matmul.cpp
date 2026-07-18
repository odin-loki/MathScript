#include "ms/distributed/matmul.hpp"
#include "ms/linalg/linalg.hpp"

#if defined(MS_HAS_MPI) && MS_HAS_MPI
#include <mpi.h>
#endif

namespace ms::distributed {

namespace {

template<typename S, template<typename> class Alloc>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> stub_gather_matmul(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& B,
    MPIContext& ctx) {
    auto global_A = gather(A, ctx);
    if (!global_A) {
        return std::unexpected(global_A.error());
    }
    auto global_B = gather(B, ctx);
    if (!global_B) {
        return std::unexpected(global_B.error());
    }
    return ms::matmul(*global_A, *global_B);
}

#if defined(MS_HAS_MPI) && MS_HAS_MPI
template<typename S, template<typename> class Alloc>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> broadcast_matrix(
    const Matrix<S, StorageOrder::ColMajor, Alloc>& root,
    MPIContext& ctx) {
    int rows = 0;
    int cols = 0;
    if (rank(ctx) == 0) {
        rows = static_cast<int>(root.rows());
        cols = static_cast<int>(root.cols());
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    Matrix<S, StorageOrder::ColMajor, Alloc> out(
        static_cast<size_t>(rows), static_cast<size_t>(cols));
    if (rank(ctx) == 0) {
        out = root;
    }
    const int count = rows * cols;
    if (count > 0) {
        MPI_Bcast(out.data(), count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    return out;
}

template<typename S, template<typename> class Alloc>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> mpi_block_matmul(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& B,
    MPIContext& ctx) {
    Matrix<S, StorageOrder::ColMajor, Alloc> global_B;
    if (rank(ctx) == 0) {
        auto gathered_B = gather(B, ctx);
        if (!gathered_B) {
            return std::unexpected(gathered_B.error());
        }
        global_B = std::move(*gathered_B);
    }

    auto B_full = broadcast_matrix(global_B, ctx);
    if (!B_full) {
        return std::unexpected(B_full.error());
    }

    auto local_C = ms::matmul(A.local, *B_full);
    if (!local_C) {
        return std::unexpected(local_C.error());
    }

    DistMatrix<S, Alloc> dist_C;
    dist_C.local = std::move(*local_C);
    dist_C.global_rows = A.global_rows;
    dist_C.global_cols = B.global_cols;
    dist_C.distribution = A.distribution;
    dist_C.owner_rank = rank(ctx);
    dist_C.row_map = A.row_map;

    auto global_C = gather(dist_C, ctx);
    if (!global_C) {
        return std::unexpected(global_C.error());
    }
    if (rank(ctx) == 0) {
        return *global_C;
    }
    return dist_C.local;
}
#endif

} // namespace

template<typename S, template<typename> class Alloc>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> matmul(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& B,
    MPIContext& ctx) {
    if (A.global_cols != B.global_rows) {
        return std::unexpected(DomainError{
            "dist_matmul", "inner dimensions must match (A cols vs B rows)"});
    }

#if defined(MS_HAS_MPI) && MS_HAS_MPI
    if (size(ctx) > 1) {
        return mpi_block_matmul(A, B, ctx);
    }
#endif

    return stub_gather_matmul(A, B, ctx);
}

template Result<Matrix<double>> matmul(
    const DistMatrix<double>&, const DistMatrix<double>&, MPIContext&);

} // namespace ms::distributed
