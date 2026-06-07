#include "ms/cuda/elementwise.hpp"
#include <algorithm>

namespace ms::cuda {

void add_inplace(std::span<double> a, std::span<const double> b, double alpha) {
    const size_t n = (std::min)(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        a[i] += alpha * b[i];
    }
}

void fill(std::span<double> out, double value) {
    for (double& v : out) {
        v = value;
    }
}

} // namespace ms::cuda
