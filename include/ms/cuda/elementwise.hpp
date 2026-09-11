// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// SPDX-FileComment: links the NVIDIA CUDA libraries; see LICENSE.exceptions
#pragma once

#include <span>

namespace ms::cuda {

void add_inplace(std::span<double> a, std::span<const double> b, double alpha = 1.0);
void fill(std::span<double> out, double value);
void mul_inplace(std::span<double> a, std::span<const double> b);
void scale(std::span<double> a, double alpha);

} // namespace ms::cuda
