#include "ms/distributed/matmul.hpp"
#include "ms/distributed/dist_ops.hpp"
#include "ms/linalg/linalg.hpp"

#include <cstddef>
#include <limits>

namespace ms::distributed {

namespace {

// Below this many multiply-accumulate operations the SUMMA collectives cost
// more than gathering both operands and multiplying locally: 16^3.
constexpr size_t kSummaMinWork = 4096;

// Any global dimension below this leaves grid blocks empty or one element wide,
// so the 2D decomposition buys nothing.
constexpr size_t kSummaMinDim = 8;

// The SUMMA core is double-only because MPI_DOUBLE is the wire type, so the
// templated entry points convert element-wise. For the only instantiation
// (S = double) this is a plain O(m*n) copy, negligible against O(m*n*k).
template<typename S, template<typename> class Alloc>
Matrix<double> to_double_block(const Matrix<S, StorageOrder::ColMajor, Alloc>& m) {
    Matrix<double> out(m.rows(), m.cols());
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            out(i, j) = static_cast<double>(m(i, j));
        }
    }
    return out;
}

template<typename S, template<typename> class Alloc>
Matrix<S, StorageOrder::ColMajor, Alloc> from_double_block(const Matrix<double>& m) {
    Matrix<S, StorageOrder::ColMajor, Alloc> out(m.rows(), m.cols());
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            out(i, j) = static_cast<S>(m(i, j));
        }
    }
    return out;
}

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

// True when SUMMA can and should run. Everything here is cheap metadata: no
// element of either operand is touched.
template<typename S, template<typename> class Alloc>
bool summa_applicable(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& B,
    const MPIContext& ctx) {
    if (!mpi_available()) {
        return false;
    }
    if (size(ctx) <= 1 || !mpi_active(ctx)) {
        return false;
    }
    const ProcessGrid grid = make_process_grid(size(ctx), rank(ctx));
    if (grid.rows == 1 && grid.cols == 1) {
        return false;
    }
    if (A.distribution != Distribution::Block ||
        B.distribution != Distribution::Block) {
        return false;
    }
    const RowLayout la = make_row_layout(A.global_rows, size(ctx));
    const RowLayout lb = make_row_layout(B.global_rows, size(ctx));
    if (!layout_matches(la, rank(ctx), A.local.rows()) ||
        !layout_matches(lb, rank(ctx), B.local.rows())) {
        return false;
    }
    if (A.local.cols() != A.global_cols || B.local.cols() != B.global_cols) {
        return false;
    }
    if (A.global_rows < kSummaMinDim || A.global_cols < kSummaMinDim ||
        B.global_cols < kSummaMinDim) {
        return false;
    }

    // M*K*N with saturating arithmetic: an overflow means the product is far
    // above the threshold, so it must not be read as a false negative.
    constexpr size_t cap = std::numeric_limits<size_t>::max();
    if (A.global_cols != 0 && A.global_rows > cap / A.global_cols) {
        return true;
    }
    const size_t mk = A.global_rows * A.global_cols;
    if (B.global_cols != 0 && mk > cap / B.global_cols) {
        return true;
    }
    return mk * B.global_cols >= kSummaMinWork;
}

} // namespace

template<typename S, template<typename> class Alloc>
Result<DistMatrix<S, Alloc>> matmul_rowblock(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& B,
    MPIContext& ctx) {
    if (A.global_cols != B.global_rows) {
        return std::unexpected(DimensionMismatch{A.global_cols, B.global_rows});
    }
    if (A.distribution != Distribution::Block ||
        B.distribution != Distribution::Block) {
        return std::unexpected(DomainError{
            "dist_matmul_summa", "requires Distribution::Block on both operands"});
    }

    const RowLayout la = make_row_layout(A.global_rows, size(ctx));
    const RowLayout lb = make_row_layout(B.global_rows, size(ctx));
    if (!layout_matches(la, rank(ctx), A.local.rows()) ||
        !layout_matches(lb, rank(ctx), B.local.rows()) ||
        A.local.cols() != A.global_cols || B.local.cols() != B.global_cols) {
        return std::unexpected(DomainError{
            "dist_matmul_summa", "local block does not match the block row layout"});
    }

    auto c_local = summa_matmul_rowblock(
        ctx,
        to_double_block(A.local), A.global_rows, A.global_cols,
        to_double_block(B.local), B.global_rows, B.global_cols);
    if (!c_local) {
        return std::unexpected(c_local.error());
    }

    DistMatrix<S, Alloc> out;
    out.local = from_double_block<S, Alloc>(*c_local);
    out.distribution = Distribution::Block;
    out.owner_rank = rank(ctx);
    out.global_rows = A.global_rows;
    out.global_cols = B.global_cols;
    return out;
}

template<typename S, template<typename> class Alloc>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> matmul_summa(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& B,
    MPIContext& ctx) {
    auto c_dist = matmul_rowblock(A, B, ctx);
    if (!c_dist) {
        return std::unexpected(c_dist.error());
    }
    const RowLayout lc = make_row_layout(A.global_rows, size(ctx));
    auto c_full = dist_allgather_rows(ctx, to_double_block(c_dist->local), lc);
    if (!c_full) {
        return std::unexpected(c_full.error());
    }
    return from_double_block<S, Alloc>(*c_full);
}

template<typename S, template<typename> class Alloc>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> matmul(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& B,
    MPIContext& ctx) {
    if (A.global_cols != B.global_rows) {
        return std::unexpected(DomainError{
            "dist_matmul", "inner dimensions must match (A cols vs B rows)"});
    }
    if (!summa_applicable(A, B, ctx)) {
        return stub_gather_matmul(A, B, ctx);
    }
    return matmul_summa(A, B, ctx);
}

template Result<Matrix<double>> matmul(
    const DistMatrix<double>&, const DistMatrix<double>&, MPIContext&);
template Result<Matrix<double>> matmul_summa(
    const DistMatrix<double>&, const DistMatrix<double>&, MPIContext&);
template Result<DistMatrix<double>> matmul_rowblock(
    const DistMatrix<double>&, const DistMatrix<double>&, MPIContext&);

} // namespace ms::distributed
