#include "ms/core/operations.hpp"

namespace ms {

template<typename S, StorageOrder Order, template<typename> class Alloc>
Matrix<S, StorageOrder::RowMajor, Alloc> transpose(const Matrix<S, Order, Alloc>& A) {
    Matrix<S, StorageOrder::RowMajor, Alloc> T(A.cols(), A.rows());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            T(j, i) = A(i, j);
        }
    }
    return T;
}

template Matrix<double, StorageOrder::RowMajor> transpose(const Matrix<double>&);
template Matrix<float, StorageOrder::RowMajor> transpose(const Matrix<float>&);

} // namespace ms
