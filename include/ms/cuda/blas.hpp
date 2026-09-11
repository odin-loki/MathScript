// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// SPDX-FileComment: links the NVIDIA CUDA libraries; see LICENSE.exceptions
#pragma once

#include "ms/core/matrix.hpp"
#include "ms/error/error_types.hpp"

namespace ms::cuda {

Result<Matrix<double>> matmul(
    const Matrix<double, StorageOrder::ColMajor>& A,
    const Matrix<double, StorageOrder::ColMajor>& B,
    int device = 0);

} // namespace ms::cuda
