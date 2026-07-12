#include "ms/core/sparse.hpp"

#include <utility>

namespace ms {

template<typename S, template<typename> class Alloc>
Sparse<S, Alloc>::Sparse(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), nnz_(0) {}

template<typename S, template<typename> class Alloc>
Sparse<S, Alloc>::Sparse(size_t rows, size_t cols, std::vector<IndexType> row,
                         std::vector<IndexType> col, std::vector<S> values)
    : rows_(rows),
      cols_(cols),
      rowidx_(std::move(row)),
      colind_(std::move(col)),
      values_(std::move(values)) {
    nnz_ = values_.size();
}

template<typename S, template<typename> class Alloc>
Matrix<S, StorageOrder::ColMajor, Alloc> Sparse<S, Alloc>::to_dense() const {
    Matrix<S, StorageOrder::ColMajor, Alloc> dense(rows_, cols_);
    for (size_t i = 0; i < values_.size(); ++i) {
        dense(rowidx_[i], colind_[i]) = values_[i];
    }
    return dense;
}

template<typename S, template<typename> class Alloc>
Matrix<S, StorageOrder::ColMajor, Alloc> Sparse<S, Alloc>::spmv(
    const Matrix<S, StorageOrder::ColMajor, Alloc>& x) const {
    Matrix<S, StorageOrder::ColMajor, Alloc> y(rows_, 1, S(0));
    for (size_t k = 0; k < values_.size(); ++k) {
        y(rowidx_[k], 0) += values_[k] * x(colind_[k], 0);
    }
    return y;
}

template<typename S, template<typename> class Alloc>
Sparse<S, Alloc> sparse_add(const Sparse<S, Alloc>& a, const Sparse<S, Alloc>& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        return Sparse<S, Alloc>(0, 0);
    }

    using IndexType = typename Sparse<S, Alloc>::IndexType;
    std::vector<IndexType> rows;
    std::vector<IndexType> cols;
    std::vector<S> vals;

    const size_t na = a.values_.size();
    const size_t nb = b.values_.size();
    size_t i = 0;
    size_t j = 0;

    auto coo_less_entries = [&](const Sparse<S, Alloc>& lhs, size_t li,
                                const Sparse<S, Alloc>& rhs, size_t ri) {
        if (lhs.rowidx_[li] != rhs.rowidx_[ri]) return lhs.rowidx_[li] < rhs.rowidx_[ri];
        return lhs.colind_[li] < rhs.colind_[ri];
    };

    auto accumulate_at = [&](const Sparse<S, Alloc>& m, size_t& idx, size_t row, size_t col) {
        S sum = S(0);
        const size_t n = m.values_.size();
        while (idx < n && m.rowidx_[idx] == row && m.colind_[idx] == col) {
            sum += m.values_[idx++];
        }
        return sum;
    };

    while (i < na || j < nb) {
        if (j >= nb || (i < na && coo_less_entries(a, i, b, j))) {
            const size_t row = a.rowidx_[i];
            const size_t col = a.colind_[i];
            const S sum = accumulate_at(a, i, row, col);
            if (sum != S(0)) {
                rows.push_back(row);
                cols.push_back(col);
                vals.push_back(sum);
            }
        } else if (i >= na || coo_less_entries(b, j, a, i)) {
            const size_t row = b.rowidx_[j];
            const size_t col = b.colind_[j];
            const S sum = accumulate_at(b, j, row, col);
            if (sum != S(0)) {
                rows.push_back(row);
                cols.push_back(col);
                vals.push_back(sum);
            }
        } else {
            const size_t row = a.rowidx_[i];
            const size_t col = a.colind_[i];
            const S sum = accumulate_at(a, i, row, col) + accumulate_at(b, j, row, col);
            if (sum != S(0)) {
                rows.push_back(row);
                cols.push_back(col);
                vals.push_back(sum);
            }
        }
    }

    return Sparse<S, Alloc>(a.rows(), a.cols(), std::move(rows), std::move(cols), std::move(vals));
}

template class Sparse<double>;
template class Sparse<float>;

template Sparse<double, std::allocator> sparse_add<double, std::allocator>(
    const Sparse<double, std::allocator>&, const Sparse<double, std::allocator>&);
template Sparse<float, std::allocator> sparse_add<float, std::allocator>(
    const Sparse<float, std::allocator>&, const Sparse<float, std::allocator>&);

} // namespace ms
