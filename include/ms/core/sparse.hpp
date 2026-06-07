#pragma once

#include <vector>

#include "ms/core/matrix.hpp"

namespace ms {

template<typename S, template<typename> class Alloc = std::allocator>
class Sparse {
public:
    using value_type = S;
    using IndexType = size_t;

    Sparse(size_t rows, size_t cols);
    Sparse(size_t rows, size_t cols, std::vector<IndexType> row,
           std::vector<IndexType> col, std::vector<S> values);

    size_t nnz() const { return nnz_; }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    Matrix<S, StorageOrder::ColMajor, Alloc> to_dense() const;
    Matrix<S, StorageOrder::ColMajor, Alloc> spmv(
        const Matrix<S, StorageOrder::ColMajor, Alloc>& x) const;

private:
    size_t rows_;
    size_t cols_;
    size_t nnz_;
    std::vector<S> values_;
    std::vector<IndexType> rowidx_;
    std::vector<IndexType> colind_;
};

} // namespace ms
