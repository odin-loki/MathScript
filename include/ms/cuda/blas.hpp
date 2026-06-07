#pragma once

#include "ms/core/matrix.hpp"
#include "ms/error/error_types.hpp"

namespace ms::cuda {

Result<Matrix<double>> matmul(
    const Matrix<double, StorageOrder::ColMajor>& A,
    const Matrix<double, StorageOrder::ColMajor>& B,
    int device = 0);

} // namespace ms::cuda
