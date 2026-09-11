// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// MathScript Transpose Header

#pragma once

#include "ms/core/matrix.hpp"
#include "ms/memory/aligned_allocator.hpp"

namespace ms {

template<typename S, StorageOrder Order, template<typename> class Alloc>
Matrix<S, StorageOrder::RowMajor, memory::AlignedAllocator>
transpose(const Matrix<S, Order, Alloc>& A);

} // namespace ms