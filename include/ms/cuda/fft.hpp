// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// SPDX-FileComment: links the NVIDIA CUDA libraries; see LICENSE.exceptions
#pragma once

#include <complex>
#include <vector>
#include "ms/error/error_types.hpp"

namespace ms::cuda {

Result<std::vector<std::complex<double>>> fft(const std::vector<double>& x);
Result<std::vector<double>> ifft(const std::vector<std::complex<double>>& x);

} // namespace ms::cuda
