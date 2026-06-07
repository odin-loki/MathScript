#pragma once

#include "ms/core/matrix.hpp"
#include "ms/error/error_types.hpp"
#include <tuple>

namespace ms::cuda {

Result<std::tuple<Matrix<double>, Matrix<double>, Matrix<double>>> lu(
    const Matrix<double, StorageOrder::ColMajor>& A,
    int device = 0);

Result<Matrix<double>> solve(
    const Matrix<double, StorageOrder::ColMajor>& A,
    const Matrix<double, StorageOrder::ColMajor>& b,
    int device = 0);

} // namespace ms::cuda
