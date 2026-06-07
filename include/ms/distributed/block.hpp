#pragma once

#include <cstddef>
#include <utility>

namespace ms::distributed {

struct RowExtent {
    size_t start = 0;
    size_t count = 0;
};

inline RowExtent block_row_extent(size_t global_rows, int rank, int nprocs) {
    if (nprocs <= 0 || global_rows == 0) {
        return {};
    }
    const size_t p = static_cast<size_t>(nprocs);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = global_rows / p;
    const size_t rem = global_rows % p;
    RowExtent ext;
    ext.start = r * base + (r < rem ? r : rem);
    ext.count = base + (r < rem ? 1 : 0);
    return ext;
}

inline std::vector<size_t> block_cyclic_row_indices(
    size_t global_rows,
    int rank,
    int nprocs) {
    std::vector<size_t> indices;
    if (nprocs <= 0 || global_rows == 0) {
        return indices;
    }
    for (size_t i = static_cast<size_t>(rank); i < global_rows;
         i += static_cast<size_t>(nprocs)) {
        indices.push_back(i);
    }
    return indices;
}

} // namespace ms::distributed
