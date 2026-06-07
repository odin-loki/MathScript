#include "ms/core/sparse.hpp"

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

template class Sparse<double>;
template class Sparse<float>;

} // namespace ms
