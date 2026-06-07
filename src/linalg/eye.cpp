#include "ms/core/operations.hpp"

namespace ms {

template<typename S, template<typename> class Alloc>
Matrix<S, StorageOrder::ColMajor, Alloc> eye(size_t n) {
    Matrix<S, StorageOrder::ColMajor, Alloc> I(n, n, S(0));
    for (size_t i = 0; i < n; ++i) {
        I(i, i) = S(1);
    }
    return I;
}

template Matrix<double> eye<double>(size_t);
template Matrix<float> eye<float>(size_t);

} // namespace ms
