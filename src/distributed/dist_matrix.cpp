#include "ms/distributed/dist_matrix.hpp"
#include "ms/distributed/block.hpp"
#include <vector>

#if defined(MS_HAS_MPI) && MS_HAS_MPI
#include <mpi.h>
#endif

namespace ms::distributed {

namespace {

template<typename S, template<typename> class Alloc>
Matrix<S, StorageOrder::ColMajor, Alloc> extract_block(
    const Matrix<S, StorageOrder::ColMajor, Alloc>& global,
    const RowExtent& ext) {
    Matrix<S, StorageOrder::ColMajor, Alloc> local(ext.count, global.cols());
    for (size_t i = 0; i < ext.count; ++i) {
        for (size_t j = 0; j < global.cols(); ++j) {
            local(i, j) = global(ext.start + i, j);
        }
    }
    return local;
}

template<typename S, template<typename> class Alloc>
Matrix<S, StorageOrder::ColMajor, Alloc> extract_cyclic(
    const Matrix<S, StorageOrder::ColMajor, Alloc>& global,
    const std::vector<size_t>& row_map) {
    Matrix<S, StorageOrder::ColMajor, Alloc> local(row_map.size(), global.cols());
    for (size_t i = 0; i < row_map.size(); ++i) {
        for (size_t j = 0; j < global.cols(); ++j) {
            local(i, j) = global(row_map[i], j);
        }
    }
    return local;
}

} // namespace

template<typename S, template<typename> class Alloc>
Result<DistMatrix<S, Alloc>> scatter(
    const Matrix<S, StorageOrder::ColMajor, Alloc>& global,
    MPIContext& ctx,
    Distribution distribution) {
    DistMatrix<S, Alloc> dist;
    dist.global_rows = global.rows();
    dist.global_cols = global.cols();
    dist.owner_rank = rank(ctx);
    dist.distribution = distribution;

    if (distribution == Distribution::BlockCyclic) {
        dist.row_map = block_cyclic_row_indices(global.rows(), rank(ctx), size(ctx));
        dist.local = extract_cyclic(global, dist.row_map);
        return dist;
    }

    if (distribution != Distribution::Block) {
        dist.local = global;
        return dist;
    }

    const RowExtent ext = block_row_extent(global.rows(), rank(ctx), size(ctx));
    dist.local = extract_block(global, ext);
    return dist;
}

template<typename S, template<typename> class Alloc>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> gather(
    const DistMatrix<S, Alloc>& dist,
    MPIContext& ctx) {
#if defined(MS_HAS_MPI) && MS_HAS_MPI
    const int nprocs = size(ctx);
    if (nprocs <= 1) {
        return dist.local;
    }

    if (dist.distribution == Distribution::BlockCyclic) {
        Matrix<S, StorageOrder::ColMajor, Alloc> global(dist.global_rows, dist.global_cols, S(0));
        const int local_rows = static_cast<int>(dist.local.rows());
        std::vector<int> counts(static_cast<size_t>(nprocs));
        std::vector<int> displs(static_cast<size_t>(nprocs));
        int total = 0;
        for (int r = 0; r < nprocs; ++r) {
            const auto rows = block_cyclic_row_indices(dist.global_rows, r, nprocs);
            counts[static_cast<size_t>(r)] = static_cast<int>(rows.size());
            displs[static_cast<size_t>(r)] = total;
            total += counts[static_cast<size_t>(r)];
        }

        std::vector<S> recv(static_cast<size_t>(total));
        std::vector<S> send(static_cast<size_t>(local_rows));
        for (int c = 0; c < static_cast<int>(dist.global_cols); ++c) {
            for (int i = 0; i < local_rows; ++i) {
                send[static_cast<size_t>(i)] = dist.local(static_cast<size_t>(i), static_cast<size_t>(c));
            }
            MPI_Gatherv(
                send.data(),
                local_rows,
                MPI_DOUBLE,
                recv.data(),
                counts.data(),
                displs.data(),
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);
            if (rank(ctx) == 0) {
                for (int r = 0; r < nprocs; ++r) {
                    const auto rows = block_cyclic_row_indices(dist.global_rows, r, nprocs);
                    for (size_t i = 0; i < rows.size(); ++i) {
                        global(rows[i], static_cast<size_t>(c)) =
                            recv[static_cast<size_t>(displs[static_cast<size_t>(r)] + static_cast<int>(i))];
                    }
                }
            }
        }

        if (rank(ctx) == 0) {
            return global;
        }
        return dist.local;
    }

    std::vector<int> counts(static_cast<size_t>(nprocs));
    std::vector<int> displs(static_cast<size_t>(nprocs));
    int total = 0;
    for (int r = 0; r < nprocs; ++r) {
        const RowExtent ext = block_row_extent(dist.global_rows, r, nprocs);
        counts[static_cast<size_t>(r)] = static_cast<int>(ext.count);
        displs[static_cast<size_t>(r)] = total;
        total += counts[static_cast<size_t>(r)];
    }

    Matrix<S, StorageOrder::ColMajor, Alloc> global(dist.global_rows, dist.global_cols);
    const int cols = static_cast<int>(dist.global_cols);

    if (rank(ctx) == 0) {
        for (size_t j = 0; j < dist.global_cols; ++j) {
            for (size_t i = 0; i < dist.local.rows(); ++i) {
                global(i, j) = dist.local(i, j);
            }
        }
    }

    std::vector<S> recv(static_cast<size_t>(total));
    std::vector<S> send(dist.local.rows());
    for (int c = 0; c < cols; ++c) {
        for (size_t i = 0; i < dist.local.rows(); ++i) {
            send[i] = dist.local(i, static_cast<size_t>(c));
        }
        MPI_Gatherv(
            send.data(),
            static_cast<int>(send.size()),
            MPI_DOUBLE,
            recv.data(),
            counts.data(),
            displs.data(),
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD);
        if (rank(ctx) == 0) {
            for (int r = 0; r < nprocs; ++r) {
                const RowExtent ext = block_row_extent(dist.global_rows, r, nprocs);
                for (int i = 0; i < counts[static_cast<size_t>(r)]; ++i) {
                    global(ext.start + static_cast<size_t>(i), static_cast<size_t>(c)) =
                        recv[static_cast<size_t>(displs[static_cast<size_t>(r)] + i)];
                }
            }
        }
    }

    if (rank(ctx) == 0) {
        return global;
    }
    return dist.local;
#else
    (void)ctx;
    Matrix<S, StorageOrder::ColMajor, Alloc> global(dist.global_rows, dist.global_cols, S(0));

    if (dist.distribution == Distribution::BlockCyclic) {
        for (size_t i = 0; i < dist.row_map.size(); ++i) {
            for (size_t j = 0; j < dist.global_cols; ++j) {
                global(dist.row_map[i], j) = dist.local(i, j);
            }
        }
        return global;
    }

    if (dist.global_rows == dist.local.rows()) {
        return dist.local;
    }

    const int nprocs = size(ctx);
    for (int r = 0; r < nprocs; ++r) {
        const RowExtent ext = block_row_extent(dist.global_rows, r, nprocs);
        if (r == rank(ctx)) {
            for (size_t i = 0; i < ext.count; ++i) {
                for (size_t j = 0; j < dist.global_cols; ++j) {
                    global(ext.start + i, j) = dist.local(i, j);
                }
            }
        }
    }
    return global;
#endif
}

template Result<DistMatrix<double>> scatter(
    const Matrix<double>&, MPIContext&, Distribution);
template Result<Matrix<double>> gather(
    const DistMatrix<double>&, MPIContext&);

template<typename S, template<typename> class Alloc>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> combine_gather(
    const std::vector<DistMatrix<S, Alloc>>& shards) {
    if (shards.empty()) {
        return std::unexpected(DomainError{"combine_gather", "no shards"});
    }

    const auto& ref = shards.front();
    Matrix<S, StorageOrder::ColMajor, Alloc> global(ref.global_rows, ref.global_cols, S(0));
    for (const auto& shard : shards) {
        if (shard.global_rows != ref.global_rows || shard.global_cols != ref.global_cols ||
            shard.distribution != ref.distribution) {
            return std::unexpected(DomainError{"combine_gather", "inconsistent shards"});
        }

        if (shard.distribution == Distribution::BlockCyclic) {
            for (size_t i = 0; i < shard.row_map.size(); ++i) {
                for (size_t j = 0; j < shard.global_cols; ++j) {
                    global(shard.row_map[i], j) = shard.local(i, j);
                }
            }
            continue;
        }

        const RowExtent ext =
            block_row_extent(shard.global_rows, shard.owner_rank, static_cast<int>(shards.size()));
        for (size_t i = 0; i < shard.local.rows(); ++i) {
            for (size_t j = 0; j < shard.global_cols; ++j) {
                global(ext.start + i, j) = shard.local(i, j);
            }
        }
    }
    return global;
}

template Result<Matrix<double>> combine_gather(const std::vector<DistMatrix<double>>&);

} // namespace ms::distributed
