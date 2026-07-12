#pragma once

#include <vector>

#include "ms/core/matrix.hpp"

namespace ms {

template<typename S, size_t N = 2>
class Tensor {
public:
    using value_type = S;

    Tensor(const std::vector<size_t>& shape);
    Tensor(size_t dim1, size_t dim2, size_t dim3 = 0);

    size_t dims() const { return dims_; }
    size_t size(size_t dim) const;
    size_t total_size() const;

    value_type& at(size_t i, size_t j) { return data_[i * shape_[1] + j]; }
    const value_type& at(size_t i, size_t j) const { return data_[i * shape_[1] + j]; }

    Tensor reshape(const std::vector<size_t>& new_shape) const;

private:
    size_t dims_;
    std::vector<size_t> shape_;
    std::vector<S> data_;
};

} // namespace ms
