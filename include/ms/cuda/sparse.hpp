// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// SPDX-FileComment: links the NVIDIA CUDA libraries; see LICENSE.exceptions
#pragma once

#include "ms/core/matrix.hpp"
#include "ms/error/error_types.hpp"
#include <vector>

namespace ms::cuda {

Result<Matrix<double>> spmv_coo(
    size_t rows,
    size_t cols,
    const std::vector<size_t>& row,
    const std::vector<size_t>& col,
    const std::vector<double>& values,
    const Matrix<double, StorageOrder::ColMajor>& x,
    int device = 0);

} // namespace ms::cuda
