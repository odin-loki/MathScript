#include "ms/core/tensor.hpp"

#include <stdexcept>

namespace ms {

template<typename S, size_t N>
Tensor<S, N>::Tensor(const std::vector<size_t>& shape)
    : dims_(shape.size()), shape_(shape) {
    size_t total = 1;
    for (size_t d : shape) {
        total *= d;
    }
    data_.assign(total, S(0));
}

template<typename S, size_t N>
Tensor<S, N>::Tensor(size_t dim1, size_t dim2, size_t dim3)
    : dims_(dim3 > 0 ? 3 : 2), shape_({dim1, dim2, dim3 > 0 ? dim3 : 1}) {
    data_.assign(total_size(), S(0));
}

template<typename S, size_t N>
size_t Tensor<S, N>::size(size_t dim) const {
    return (dim < dims_) ? shape_[dim] : 0;
}

template<typename S, size_t N>
size_t Tensor<S, N>::total_size() const {
    size_t total = 1;
    for (size_t d = 0; d < dims_; ++d) {
        total *= shape_[d];
    }
    return total;
}

template<typename S, size_t N>
Tensor<S, N> Tensor<S, N>::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_total = 1;
    for (size_t d : new_shape) {
        new_total *= d;
    }
    if (new_total != total_size()) {
        throw std::invalid_argument(
            "Tensor reshape: product of new shape must equal total_size()");
    }
    Tensor<S, N> result(new_shape);
    result.data_ = data_;
    return result;
}

template class Tensor<double, 2>;
template class Tensor<float, 2>;

} // namespace ms
