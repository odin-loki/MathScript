// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "ms/distributed/dist_ops.hpp"
#include "ms/cpu/blas.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#if defined(MS_HAS_MPI) && MS_HAS_MPI
#include <mpi.h>
#endif

namespace ms::distributed {

namespace {

// MPI counts are `int`; every message length is checked against this first.
constexpr std::size_t kMaxMpiCount =
    static_cast<std::size_t>(std::numeric_limits<int>::max());

// Staging buffers are never empty so that `.data()` is never null, which a rank
// owning zero rows would otherwise hand to MPI.
std::size_t buffer_size(std::size_t n) {
    return (n > 0) ? n : std::size_t{1};
}

// ---------------------------------------------------------------------------
// Guarded thin wrappers: the only functions in the module with MPI in the body.
// Each returns an MPI status code, 0 on success.
// ---------------------------------------------------------------------------

int mpi_allreduce_sum(std::vector<double>& buf) {
#if defined(MS_HAS_MPI) && MS_HAS_MPI
    std::vector<double> out(buf.size(), 0.0);
    const int rc = MPI_Allreduce(
        buf.data(), out.data(), static_cast<int>(buf.size()),
        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rc == MPI_SUCCESS) {
        buf.swap(out);
    }
    return rc;
#else
    (void)buf;
    return 0;
#endif
}

// The send buffers and count arrays are taken by NON-CONST reference on
// purpose: they are always locally owned staging vectors, and pre-MPI-3 headers
// declare the send parameters as plain `void*` / `int[]`. Binding them mutably
// is what lets this compile against both header generations without a
// const_cast.
int mpi_allgatherv(
    std::vector<double>& send,
    std::vector<double>& recv,
    std::vector<int>& counts,
    std::vector<int>& displs) {
#if defined(MS_HAS_MPI) && MS_HAS_MPI
    return MPI_Allgatherv(
        send.data(), static_cast<int>(send.size()), MPI_DOUBLE,
        recv.data(), counts.data(), displs.data(), MPI_DOUBLE,
        MPI_COMM_WORLD);
#else
    (void)counts;
    (void)displs;
    recv = send;
    return 0;
#endif
}

int mpi_alltoallv(
    std::vector<double>& send,
    std::vector<int>& send_counts,
    std::vector<int>& send_displs,
    std::vector<double>& recv,
    std::vector<int>& recv_counts,
    std::vector<int>& recv_displs) {
#if defined(MS_HAS_MPI) && MS_HAS_MPI
    return MPI_Alltoallv(
        send.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE,
        recv.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
        MPI_COMM_WORLD);
#else
    (void)send_counts;
    (void)send_displs;
    (void)recv_counts;
    (void)recv_displs;
    recv = send;
    return 0;
#endif
}

// Cartesian row/column sub-communicators for the SUMMA grid. The type is empty
// without MPI so that no MPI handle ever appears in a header or a struct that a
// non-MPI translation unit compiles.
struct GridComms {
#if defined(MS_HAS_MPI) && MS_HAS_MPI
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Comm row = MPI_COMM_NULL;
    MPI_Comm col = MPI_COMM_NULL;

    ~GridComms() {
        if (row != MPI_COMM_NULL) {
            MPI_Comm_free(&row);
        }
        if (col != MPI_COMM_NULL) {
            MPI_Comm_free(&col);
        }
        if (cart != MPI_COMM_NULL) {
            MPI_Comm_free(&cart);
        }
    }
#endif
    GridComms() = default;
    GridComms(const GridComms&) = delete;
    GridComms& operator=(const GridComms&) = delete;
    GridComms(GridComms&&) = delete;
    GridComms& operator=(GridComms&&) = delete;
};

// Creates the 2D Cartesian communicator and its two sub-communicators, then
// confirms that the coordinates MPI assigned really are (rank / Pc, rank % Pc).
// `reorder = 0` guarantees that, but an implementation that ignored it would
// silently invalidate every panel owner computed from ProcessGrid, so it is
// checked rather than assumed.
int grid_comms_create(const ProcessGrid& grid, GridComms& out) {
#if defined(MS_HAS_MPI) && MS_HAS_MPI
    int dims[2] = {grid.rows, grid.cols};
    int periods[2] = {0, 0};
    int rc = MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &out.cart);
    if (rc != MPI_SUCCESS) {
        return rc;
    }
    int coords[2] = {0, 0};
    rc = MPI_Cart_coords(out.cart, grid.rank, 2, coords);
    if (rc != MPI_SUCCESS) {
        return rc;
    }
    if (coords[0] != grid.my_row || coords[1] != grid.my_col) {
        return -1;
    }
    int remain_row[2] = {0, 1};  // vary the column index -> process ROW comm
    rc = MPI_Cart_sub(out.cart, remain_row, &out.row);
    if (rc != MPI_SUCCESS) {
        return rc;
    }
    int remain_col[2] = {1, 0};  // vary the row index -> process COLUMN comm
    return MPI_Cart_sub(out.cart, remain_col, &out.col);
#else
    (void)grid;
    (void)out;
    return 0;
#endif
}

int bcast_row(GridComms& comms, std::vector<double>& buf, int root) {
#if defined(MS_HAS_MPI) && MS_HAS_MPI
    if (buf.empty()) {
        return 0;
    }
    return MPI_Bcast(
        buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, root, comms.row);
#else
    (void)comms;
    (void)buf;
    (void)root;
    return 0;
#endif
}

int bcast_col(GridComms& comms, std::vector<double>& buf, int root) {
#if defined(MS_HAS_MPI) && MS_HAS_MPI
    if (buf.empty()) {
        return 0;
    }
    return MPI_Bcast(
        buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, root, comms.col);
#else
    (void)comms;
    (void)buf;
    (void)root;
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Local helpers with no MPI in them at all.
// ---------------------------------------------------------------------------

// Which of `nparts` block-row parts of `n_global` owns global index `k`.
int block_owner(std::size_t k, std::size_t n_global, int nparts) {
    const int parts = (nparts > 0) ? nparts : 1;
    for (int r = 0; r < parts; ++r) {
        const RowExtent e = block_row_extent(n_global, r, parts);
        if (e.count > 0 && k >= e.start && k < e.start + e.count) {
            return r;
        }
    }
    return 0;
}

// One rank's rectangular piece of a global matrix.
struct BlockExtent {
    std::size_t row_start = 0;
    std::size_t row_count = 0;
    std::size_t col_start = 0;
    std::size_t col_count = 0;
};

// Overlap of the half-open ranges [a_start, a_start + a_count) and
// [b_start, b_start + b_count).
RowExtent intersect(
    std::size_t a_start, std::size_t a_count,
    std::size_t b_start, std::size_t b_count) {
    const std::size_t lo = std::max(a_start, b_start);
    const std::size_t hi = std::min(a_start + a_count, b_start + b_count);
    RowExtent out;
    out.start = lo;
    out.count = (hi > lo) ? (hi - lo) : std::size_t{0};
    return out;
}

// Move data from the caller's `src_map[rank]` extent to its `dst_map[rank]`
// extent with a single MPI_Alltoallv. Sub-blocks travel packed column-major of
// the intersection, so an element at intersection offset (ii, jj) sits at
// jj * rows + ii. With one rank the pack/unpack pair is an exact identity copy.
Result<Matrix<double>> redistribute_blocks(
    const MPIContext& ctx,
    const Matrix<double>& src_local,
    const std::vector<BlockExtent>& src_map,
    const std::vector<BlockExtent>& dst_map) {
    const int my_rank = rank(ctx);
    const int nprocs = (size(ctx) > 0) ? size(ctx) : 1;
    const std::size_t procs = static_cast<std::size_t>(nprocs);
    if (src_map.size() != procs || dst_map.size() != procs ||
        my_rank < 0 || my_rank >= nprocs) {
        return std::unexpected(DomainError{
            "redistribute_blocks", "process map does not cover this context"});
    }

    const std::size_t me = static_cast<std::size_t>(my_rank);
    const BlockExtent src = src_map[me];
    const BlockExtent dst = dst_map[me];
    if (src_local.rows() != src.row_count || src_local.cols() != src.col_count) {
        return std::unexpected(
            DimensionMismatch{src_local.rows(), src.row_count});
    }

    Matrix<double> out(dst.row_count, dst.col_count, 0.0);

    if (!mpi_active(ctx) || nprocs <= 1) {
        const RowExtent ri =
            intersect(src.row_start, src.row_count, dst.row_start, dst.row_count);
        const RowExtent ci =
            intersect(src.col_start, src.col_count, dst.col_start, dst.col_count);
        for (std::size_t jj = 0; jj < ci.count; ++jj) {
            for (std::size_t ii = 0; ii < ri.count; ++ii) {
                out(ri.start - dst.row_start + ii, ci.start - dst.col_start + jj) =
                    src_local(ri.start - src.row_start + ii,
                              ci.start - src.col_start + jj);
            }
        }
        return out;
    }

    std::vector<int> send_counts(procs, 0);
    std::vector<int> send_displs(procs, 0);
    std::vector<int> recv_counts(procs, 0);
    std::vector<int> recv_displs(procs, 0);

    std::size_t send_total = 0;
    std::size_t recv_total = 0;
    for (std::size_t r = 0; r < procs; ++r) {
        const RowExtent sri = intersect(
            src.row_start, src.row_count, dst_map[r].row_start, dst_map[r].row_count);
        const RowExtent sci = intersect(
            src.col_start, src.col_count, dst_map[r].col_start, dst_map[r].col_count);
        const std::size_t sc = sri.count * sci.count;

        const RowExtent rri = intersect(
            dst.row_start, dst.row_count, src_map[r].row_start, src_map[r].row_count);
        const RowExtent rci = intersect(
            dst.col_start, dst.col_count, src_map[r].col_start, src_map[r].col_count);
        const std::size_t rc = rri.count * rci.count;

        if (sc > kMaxMpiCount || send_total > kMaxMpiCount - sc ||
            rc > kMaxMpiCount || recv_total > kMaxMpiCount - rc) {
            return std::unexpected(DomainError{
                "redistribute_blocks", "element count exceeds MPI int range"});
        }
        send_counts[r] = static_cast<int>(sc);
        send_displs[r] = static_cast<int>(send_total);
        send_total += sc;
        recv_counts[r] = static_cast<int>(rc);
        recv_displs[r] = static_cast<int>(recv_total);
        recv_total += rc;
    }

    std::vector<double> send(buffer_size(send_total), 0.0);
    for (std::size_t r = 0; r < procs; ++r) {
        const RowExtent ri = intersect(
            src.row_start, src.row_count, dst_map[r].row_start, dst_map[r].row_count);
        const RowExtent ci = intersect(
            src.col_start, src.col_count, dst_map[r].col_start, dst_map[r].col_count);
        const std::size_t base = static_cast<std::size_t>(send_displs[r]);
        for (std::size_t jj = 0; jj < ci.count; ++jj) {
            for (std::size_t ii = 0; ii < ri.count; ++ii) {
                send[base + jj * ri.count + ii] = src_local(
                    ri.start - src.row_start + ii, ci.start - src.col_start + jj);
            }
        }
    }

    std::vector<double> recv(buffer_size(recv_total), 0.0);
    const int rc = mpi_alltoallv(
        send, send_counts, send_displs, recv, recv_counts, recv_displs);
    if (rc != 0) {
        return std::unexpected(DistributedError{my_rank, rc});
    }

    for (std::size_t r = 0; r < procs; ++r) {
        const RowExtent ri = intersect(
            dst.row_start, dst.row_count, src_map[r].row_start, src_map[r].row_count);
        const RowExtent ci = intersect(
            dst.col_start, dst.col_count, src_map[r].col_start, src_map[r].col_count);
        const std::size_t base = static_cast<std::size_t>(recv_displs[r]);
        for (std::size_t jj = 0; jj < ci.count; ++jj) {
            for (std::size_t ii = 0; ii < ri.count; ++ii) {
                out(ri.start - dst.row_start + ii, ci.start - dst.col_start + jj) =
                    recv[base + jj * ri.count + ii];
            }
        }
    }
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Layout descriptors
// ---------------------------------------------------------------------------

RowLayout make_row_layout(std::size_t global_rows, int nprocs) {
    RowLayout layout;
    layout.global_rows = global_rows;
    layout.nprocs = (nprocs > 0) ? nprocs : 1;

    const std::size_t p = static_cast<std::size_t>(layout.nprocs);
    layout.starts.assign(p, 0);
    layout.counts.assign(p, 0);

    const std::size_t base = global_rows / p;
    const std::size_t rem = global_rows % p;
    for (std::size_t r = 0; r < p; ++r) {
        layout.starts[r] = r * base + ((r < rem) ? r : rem);
        layout.counts[r] = base + ((r < rem) ? std::size_t{1} : std::size_t{0});
    }
    return layout;
}

bool layout_matches(const RowLayout& layout, int my_rank, std::size_t local_rows) {
    if (my_rank < 0 || my_rank >= layout.nprocs) {
        return false;
    }
    const std::size_t r = static_cast<std::size_t>(my_rank);
    if (r >= layout.counts.size()) {
        return false;
    }
    return layout.counts[r] == local_rows;
}

ProcessGrid make_process_grid(int nprocs, int rank_id) {
    ProcessGrid grid;
    grid.nprocs = (nprocs > 0) ? nprocs : 1;

    int best = 1;
    for (int d = 1; d <= grid.nprocs / d; ++d) {
        if (grid.nprocs % d == 0) {
            best = d;
        }
    }
    grid.rows = best;
    grid.cols = grid.nprocs / best;
    grid.rank = (rank_id >= 0 && rank_id < grid.nprocs) ? rank_id : 0;
    grid.my_row = grid.rank / grid.cols;
    grid.my_col = grid.rank % grid.cols;
    return grid;
}

std::vector<std::size_t> summa_k_breaks(
    std::size_t k_global, int grid_rows, int grid_cols) {
    std::vector<std::size_t> breaks;
    breaks.push_back(0);
    if (k_global == 0) {
        return breaks;
    }

    const int pr_n = (grid_rows > 0) ? grid_rows : 1;
    const int pc_n = (grid_cols > 0) ? grid_cols : 1;
    for (int pc = 0; pc < pc_n; ++pc) {
        breaks.push_back(block_row_extent(k_global, pc, pc_n).start);
    }
    for (int pr = 0; pr < pr_n; ++pr) {
        breaks.push_back(block_row_extent(k_global, pr, pr_n).start);
    }
    breaks.push_back(k_global);

    std::sort(breaks.begin(), breaks.end());
    breaks.erase(std::unique(breaks.begin(), breaks.end()), breaks.end());
    while (!breaks.empty() && breaks.back() > k_global) {
        breaks.pop_back();
    }
    return breaks;
}

bool mpi_available() {
#if defined(MS_HAS_MPI) && MS_HAS_MPI
    return true;
#else
    return false;
#endif
}

bool mpi_active(const MPIContext& ctx) {
    return mpi_available() && ctx.active && size(ctx) > 1;
}

// ---------------------------------------------------------------------------
// Level 1 — purely local
// ---------------------------------------------------------------------------

double local_dot(const DistVec& x, const DistVec& y) {
    double sum = 0.0;
    const std::size_t n = std::min(x.rows(), y.rows());
    for (std::size_t i = 0; i < n; ++i) {
        sum += x(i, 0) * y(i, 0);
    }
    return sum;
}

DistVec dist_axpy(double alpha, const DistVec& x, const DistVec& y) {
    DistVec r(x.rows(), x.cols());
    for (std::size_t i = 0; i < x.rows(); ++i) {
        for (std::size_t j = 0; j < x.cols(); ++j) {
            const double yv = (i < y.rows() && j < y.cols()) ? y(i, j) : 0.0;
            r(i, j) = alpha * x(i, j) + yv;
        }
    }
    return r;
}

void dist_axpy_inplace(double alpha, const DistVec& x, DistVec& y) {
    const std::size_t n = std::min(x.rows(), y.rows());
    const std::size_t m = std::min(x.cols(), y.cols());
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            y(i, j) = alpha * x(i, j) + y(i, j);
        }
    }
}

DistVec dist_scale(double alpha, const DistVec& x) {
    DistVec r(x.rows(), x.cols());
    for (std::size_t i = 0; i < x.rows(); ++i) {
        for (std::size_t j = 0; j < x.cols(); ++j) {
            r(i, j) = alpha * x(i, j);
        }
    }
    return r;
}

void dist_scal(double alpha, DistVec& x) {
    for (std::size_t i = 0; i < x.rows(); ++i) {
        for (std::size_t j = 0; j < x.cols(); ++j) {
            x(i, j) = alpha * x(i, j);
        }
    }
}

Matrix<double> dist_copy(const Matrix<double>& x) {
    Matrix<double> out(x.rows(), x.cols());
    for (std::size_t i = 0; i < x.rows(); ++i) {
        for (std::size_t j = 0; j < x.cols(); ++j) {
            out(i, j) = x(i, j);
        }
    }
    return out;
}

DistVec local_matvec(const Matrix<double>& A, const DistVec& x_full) {
    DistVec y(A.rows(), 1, 0.0);
    const std::size_t k_max = std::min(A.cols(), x_full.rows());
    for (std::size_t i = 0; i < A.rows(); ++i) {
        double sum = 0.0;
        for (std::size_t k = 0; k < k_max; ++k) {
            sum += A(i, k) * x_full(k, 0);
        }
        y(i, 0) = sum;
    }
    return y;
}

DistVec local_segment(const DistVec& full, const RowLayout& layout, int my_rank) {
    if (my_rank < 0 || my_rank >= layout.nprocs) {
        return DistVec(0, 1);
    }
    const std::size_t r = static_cast<std::size_t>(my_rank);
    if (r >= layout.counts.size()) {
        return DistVec(0, 1);
    }
    const std::size_t start = layout.starts[r];
    const std::size_t count = layout.counts[r];
    if (start + count > full.rows()) {
        return DistVec(0, 1);
    }
    DistVec out(count, 1);
    for (std::size_t i = 0; i < count; ++i) {
        out(i, 0) = full(start + i, 0);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Level 1 — collective
// ---------------------------------------------------------------------------

double dist_dot(const MPIContext& ctx, const DistVec& x, const DistVec& y) {
    const double partial = local_dot(x, y);
    if (!mpi_active(ctx)) {
        return partial;
    }
    return allreduce_sum(ctx, partial);
}

double dist_norm(const MPIContext& ctx, const DistVec& x) {
    return std::sqrt(dist_dot(ctx, x, x));
}

bool dist_all_true(const MPIContext& ctx, bool local_value) {
    const double flag = local_value ? 1.0 : 0.0;
    if (!mpi_active(ctx)) {
        return local_value;
    }
    return allreduce_min(ctx, flag) > 0.5;
}

// ---------------------------------------------------------------------------
// Level 2 — collective
// ---------------------------------------------------------------------------

Result<Matrix<double>> dist_allgather_rows(
    const MPIContext& ctx,
    const Matrix<double>& local,
    const RowLayout& layout) {
    const int my_rank = rank(ctx);
    const int nprocs = size(ctx);
    if (!layout_matches(layout, my_rank, local.rows())) {
        return std::unexpected(
            DimensionMismatch{local.rows(), layout.global_rows});
    }
    if (!mpi_active(ctx) || nprocs <= 1) {
        return dist_copy(local);
    }

    const std::size_t cols = local.cols();
    const std::size_t procs = static_cast<std::size_t>(nprocs);
    std::vector<int> counts(procs, 0);
    std::vector<int> displs(procs, 0);
    std::size_t total = 0;
    for (std::size_t r = 0; r < procs; ++r) {
        const std::size_t c = layout.counts[r] * cols;
        if (c > kMaxMpiCount || total > kMaxMpiCount - c) {
            return std::unexpected(DomainError{
                "dist_allgather_rows", "element count exceeds MPI int range"});
        }
        counts[r] = static_cast<int>(c);
        displs[r] = static_cast<int>(total);
        total += c;
    }

    std::vector<double> send(buffer_size(local.rows() * cols), 0.0);
    for (std::size_t j = 0; j < cols; ++j) {
        for (std::size_t i = 0; i < local.rows(); ++i) {
            send[j * local.rows() + i] = local(i, j);
        }
    }
    send.resize(local.rows() * cols);

    std::vector<double> recv(buffer_size(total), 0.0);
    const int rc = mpi_allgatherv(send, recv, counts, displs);
    if (rc != 0) {
        return std::unexpected(DistributedError{my_rank, rc});
    }

    Matrix<double> out(layout.global_rows, cols, 0.0);
    for (std::size_t r = 0; r < procs; ++r) {
        const std::size_t row_start = layout.starts[r];
        const std::size_t row_count = layout.counts[r];
        const std::size_t base = static_cast<std::size_t>(displs[r]);
        for (std::size_t j = 0; j < cols; ++j) {
            for (std::size_t i = 0; i < row_count; ++i) {
                out(row_start + i, j) = recv[base + j * row_count + i];
            }
        }
    }
    return out;
}

Result<DistVec> dist_matvec(
    const MPIContext& ctx,
    const Matrix<double>& A_local,
    const DistVec& x_local,
    const RowLayout& col_layout) {
    if (A_local.cols() != col_layout.global_rows) {
        return std::unexpected(
            DimensionMismatch{A_local.cols(), col_layout.global_rows});
    }
    if (!layout_matches(col_layout, rank(ctx), x_local.rows())) {
        return std::unexpected(
            DimensionMismatch{x_local.rows(), col_layout.global_rows});
    }

    auto x_full = dist_allgather_rows(ctx, x_local, col_layout);
    if (!x_full) {
        return std::unexpected(x_full.error());
    }
    return local_matvec(A_local, *x_full);
}

Result<DistVec> dist_sum_vector(const MPIContext& ctx, const DistVec& partial) {
    if (!mpi_active(ctx)) {
        return dist_copy(partial);
    }
    const std::size_t n = partial.rows();
    if (n > kMaxMpiCount) {
        return std::unexpected(DomainError{
            "dist_sum_vector", "element count exceeds MPI int range"});
    }

    std::vector<double> buf(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        buf[i] = partial(i, 0);
    }
    const int rc = mpi_allreduce_sum(buf);
    if (rc != 0) {
        return std::unexpected(DistributedError{rank(ctx), rc});
    }

    DistVec out(n, 1);
    for (std::size_t i = 0; i < n; ++i) {
        out(i, 0) = buf[i];
    }
    return out;
}

Result<DistVec> dist_matvec_transpose(
    const MPIContext& ctx,
    const Matrix<double>& A_local,
    const DistVec& u_local) {
    if (u_local.rows() < A_local.rows()) {
        return std::unexpected(DimensionMismatch{u_local.rows(), A_local.rows()});
    }

    DistVec partial(A_local.cols(), 1, 0.0);
    for (std::size_t j = 0; j < A_local.cols(); ++j) {
        double sum = 0.0;
        for (std::size_t i = 0; i < A_local.rows(); ++i) {
            sum += A_local(i, j) * u_local(i, 0);
        }
        partial(j, 0) = sum;
    }
    return dist_sum_vector(ctx, partial);
}

bool dist_is_symmetric(
    const MPIContext& ctx,
    const Matrix<double>& A_local,
    const RowLayout& layout,
    double tol) {
    const int my_rank = rank(ctx);
    const int nprocs = (size(ctx) > 0) ? size(ctx) : 1;
    if (A_local.cols() != layout.global_rows) {
        return false;
    }
    if (!layout_matches(layout, my_rank, A_local.rows())) {
        return false;
    }

    if (!mpi_active(ctx) || nprocs <= 1) {
        // Same comparison, same order and same tolerance as
        // ms::linalg_detail::is_symmetric, so dist_cg accepts exactly the
        // matrices ms::cg accepts.
        for (std::size_t i = 0; i < A_local.rows(); ++i) {
            for (std::size_t j = i + 1; j < A_local.cols(); ++j) {
                if (std::abs(A_local(i, j) - A_local(j, i)) > tol) {
                    return false;
                }
            }
        }
        return true;
    }

    const std::size_t procs = static_cast<std::size_t>(nprocs);
    const std::size_t me = static_cast<std::size_t>(my_rank);
    const std::size_t c_me = layout.counts[me];

    std::vector<int> counts(procs, 0);
    std::vector<int> displs(procs, 0);
    std::size_t total = 0;
    for (std::size_t p = 0; p < procs; ++p) {
        const std::size_t c = c_me * layout.counts[p];
        if (c > kMaxMpiCount || total > kMaxMpiCount - c) {
            return false;
        }
        counts[p] = static_cast<int>(c);
        displs[p] = static_cast<int>(total);
        total += c;
    }

    // Send peer p the sub-block A[my rows][cols of R_p] (c_me x c_p), and get
    // back A[R_p][cols of R_me] (c_p x c_me). Both are c_me * c_p doubles, so
    // the send and receive descriptors coincide.
    std::vector<double> send(buffer_size(total), 0.0);
    for (std::size_t p = 0; p < procs; ++p) {
        const std::size_t base = static_cast<std::size_t>(displs[p]);
        const std::size_t p_start = layout.starts[p];
        const std::size_t c_p = layout.counts[p];
        for (std::size_t jj = 0; jj < c_p; ++jj) {
            for (std::size_t ii = 0; ii < c_me; ++ii) {
                send[base + jj * c_me + ii] = A_local(ii, p_start + jj);
            }
        }
    }

    std::vector<double> recv(buffer_size(total), 0.0);
    const int rc = mpi_alltoallv(send, counts, displs, recv, counts, displs);
    if (rc != 0) {
        return false;
    }

    bool ok = true;
    for (std::size_t p = 0; p < procs && ok; ++p) {
        const std::size_t base = static_cast<std::size_t>(displs[p]);
        const std::size_t p_start = layout.starts[p];
        const std::size_t c_p = layout.counts[p];
        for (std::size_t jj = 0; jj < c_me && ok; ++jj) {
            for (std::size_t ii = 0; ii < c_p; ++ii) {
                if (std::abs(A_local(jj, p_start + ii) - recv[base + jj * c_p + ii]) >
                    tol) {
                    ok = false;
                    break;
                }
            }
        }
    }
    return dist_all_true(ctx, ok);
}

// ---------------------------------------------------------------------------
// Level 3 — SUMMA
// ---------------------------------------------------------------------------

Result<Matrix<double>> summa_matmul_rowblock(
    const MPIContext& ctx,
    const Matrix<double>& a_local, std::size_t a_rows, std::size_t a_cols,
    const Matrix<double>& b_local, std::size_t b_rows, std::size_t b_cols) {
    if (a_cols != b_rows) {
        return std::unexpected(DimensionMismatch{a_cols, b_rows});
    }

    const int nprocs = (size(ctx) > 0) ? size(ctx) : 1;
    const int my_rank = rank(ctx);
    const ProcessGrid grid = make_process_grid(nprocs, my_rank);
    const RowLayout la = make_row_layout(a_rows, nprocs);
    const RowLayout lb = make_row_layout(b_rows, nprocs);
    const RowLayout lc = make_row_layout(a_rows, nprocs);

    if (!layout_matches(la, my_rank, a_local.rows()) ||
        a_local.cols() != a_cols) {
        return std::unexpected(DimensionMismatch{a_local.rows(), a_rows});
    }
    if (!layout_matches(lb, my_rank, b_local.rows()) ||
        b_local.cols() != b_cols) {
        return std::unexpected(DimensionMismatch{b_local.rows(), b_rows});
    }

    // 1. Block maps: where every rank's data lives before and after each move.
    const std::size_t procs = static_cast<std::size_t>(nprocs);
    std::vector<BlockExtent> a_src(procs);
    std::vector<BlockExtent> a_dst(procs);
    std::vector<BlockExtent> b_src(procs);
    std::vector<BlockExtent> b_dst(procs);
    std::vector<BlockExtent> c_src(procs);
    std::vector<BlockExtent> c_dst(procs);
    for (std::size_t r = 0; r < procs; ++r) {
        const ProcessGrid g = make_process_grid(nprocs, static_cast<int>(r));
        const RowExtent ar = block_row_extent(a_rows, g.my_row, grid.rows);
        const RowExtent ac = block_row_extent(a_cols, g.my_col, grid.cols);
        const RowExtent br = block_row_extent(b_rows, g.my_row, grid.rows);
        const RowExtent bc = block_row_extent(b_cols, g.my_col, grid.cols);

        a_src[r] = BlockExtent{la.starts[r], la.counts[r], 0, a_cols};
        a_dst[r] = BlockExtent{ar.start, ar.count, ac.start, ac.count};
        b_src[r] = BlockExtent{lb.starts[r], lb.counts[r], 0, b_cols};
        b_dst[r] = BlockExtent{br.start, br.count, bc.start, bc.count};
        c_src[r] = BlockExtent{ar.start, ar.count, bc.start, bc.count};
        c_dst[r] = BlockExtent{lc.starts[r], lc.counts[r], 0, b_cols};
    }

    // 2. 1D block-row -> 2D block.
    auto a2d = redistribute_blocks(ctx, a_local, a_src, a_dst);
    if (!a2d) {
        return std::unexpected(a2d.error());
    }
    auto b2d = redistribute_blocks(ctx, b_local, b_src, b_dst);
    if (!b2d) {
        return std::unexpected(b2d.error());
    }

    const std::size_t m_loc = a2d->rows();  // rows of A's 2D block == C's rows
    const std::size_t n_loc = b2d->cols();  // cols of B's 2D block == C's cols
    Matrix<double> c2d(m_loc, n_loc, 0.0);

    // 3. Cartesian row and column sub-communicators.
    GridComms comms;
    const int comm_rc = grid_comms_create(grid, comms);
    if (comm_rc != 0) {
        return std::unexpected(DistributedError{my_rank, comm_rc});
    }

    // 4. Broadcast - multiply - roll. Every rank in one grid row shares m_loc
    // and every rank in one grid column shares n_loc, so the panel buffers all
    // have the same length without any extra exchange.
    const std::vector<std::size_t> breaks =
        summa_k_breaks(a_cols, grid.rows, grid.cols);
    const RowExtent my_a_cols = block_row_extent(a_cols, grid.my_col, grid.cols);
    const RowExtent my_b_rows = block_row_extent(b_rows, grid.my_row, grid.rows);

    for (std::size_t t = 0; t + 1 < breaks.size(); ++t) {
        const std::size_t k0 = breaks[t];
        const std::size_t kb = breaks[t + 1] - k0;
        const int owner_col = block_owner(k0, a_cols, grid.cols);
        const int owner_row = block_owner(k0, b_rows, grid.rows);

        std::vector<double> a_panel(buffer_size(m_loc * kb), 0.0);
        if (grid.my_col == owner_col) {
            for (std::size_t j = 0; j < kb; ++j) {
                for (std::size_t i = 0; i < m_loc; ++i) {
                    a_panel[j * m_loc + i] = (*a2d)(i, k0 - my_a_cols.start + j);
                }
            }
        }
        const int rc_a = bcast_row(comms, a_panel, owner_col);
        if (rc_a != 0) {
            return std::unexpected(DistributedError{my_rank, rc_a});
        }

        std::vector<double> b_panel(buffer_size(kb * n_loc), 0.0);
        if (grid.my_row == owner_row) {
            for (std::size_t j = 0; j < n_loc; ++j) {
                for (std::size_t i = 0; i < kb; ++i) {
                    b_panel[j * kb + i] = (*b2d)(k0 - my_b_rows.start + i, j);
                }
            }
        }
        const int rc_b = bcast_col(comms, b_panel, owner_row);
        if (rc_b != 0) {
            return std::unexpected(DistributedError{my_rank, rc_b});
        }

        // C_2d += A_panel (m_loc x kb) * B_panel (kb x n_loc). beta = 1.0 into
        // a zero-initialised C is bit-identical to a single beta = 0.0 call.
        cpu::blas::dgemm(
            'N', 'N',
            static_cast<int>(m_loc), static_cast<int>(n_loc), static_cast<int>(kb),
            1.0,
            a_panel.data(), static_cast<int>((m_loc > 0) ? m_loc : std::size_t{1}),
            b_panel.data(), static_cast<int>((kb > 0) ? kb : std::size_t{1}),
            1.0,
            c2d.data(), static_cast<int>((m_loc > 0) ? m_loc : std::size_t{1}));
    }

    // 5. 2D block -> 1D block row.
    return redistribute_blocks(ctx, c2d, c_src, c_dst);
}

} // namespace ms::distributed
