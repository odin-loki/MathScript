// MathScript Matrix Implementation

#include "ms/core/matrix.hpp"
#include "ms/error/error_types.hpp"

namespace ms {

template<typename S, StorageOrder Order, template<typename> class Alloc>
Matrix<S, Order, Alloc>::Matrix(size_t rows, size_t cols, S init)
    : m_rows_(rows), m_cols_(cols), m_data_(rows * cols, init) {}

template<typename S, StorageOrder Order, template<typename> class Alloc>
Matrix<S, Order, Alloc>::Matrix(size_t rows, size_t cols)
    : m_rows_(rows), m_cols_(cols), m_data_(rows * cols) {}

template<typename S, StorageOrder Order, template<typename> class Alloc>
size_t Matrix<S, Order, Alloc>::rows() const {
    return m_rows_;
}

template<typename S, StorageOrder Order, template<typename> class Alloc>
size_t Matrix<S, Order, Alloc>::cols() const {
    return m_cols_;
}

template<typename S, StorageOrder Order, template<typename> class Alloc>
typename Matrix<S, Order, Alloc>::reference Matrix<S, Order, Alloc>::operator()(size_t i, size_t j) {
    return m_data_[rowMajorIndex(i, j)];
}

template<typename S, StorageOrder Order, template<typename> class Alloc>
typename Matrix<S, Order, Alloc>::const_reference Matrix<S, Order, Alloc>::operator()(size_t i, size_t j) const {
    return m_data_[rowMajorIndex(i, j)];
}

template<typename S, StorageOrder Order, template<typename> class Alloc>
typename Matrix<S, Order, Alloc>::reference Matrix<S, Order, Alloc>::column(size_t j) {
    return m_data_[j];
}

template<typename S, StorageOrder Order, template<typename> class Alloc>
typename Matrix<S, Order, Alloc>::const_reference Matrix<S, Order, Alloc>::column(size_t j) const {
    return m_data_[j];
}

template<typename S, StorageOrder Order, template<typename> class Alloc>
size_t Matrix<S, Order, Alloc>::rowMajorIndex(size_t i, size_t j) {
    return i * m_cols_ + j;
}

} // namespace ms

// Explicit template instantiations
template class Matrix<double, StorageOrder::ColMajor, std::allocator>;
template class Matrix<double, StorageOrder::ColMajor, ms::memory::AlignedAllocator>;
template class Matrix<float, StorageOrder::ColMajor, std::allocator>;
template class Matrix<float, StorageOrder::ColMajor, ms::memory::AlignedAllocator>;