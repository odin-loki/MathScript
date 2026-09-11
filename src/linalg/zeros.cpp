// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "ms/core/operations.hpp"

namespace ms {

template<typename S, template<typename> class Alloc>
Matrix<S, StorageOrder::ColMajor, Alloc> zeros(size_t m, size_t n) {
    return Matrix<S, StorageOrder::ColMajor, Alloc>(m, n, S(0));
}

template Matrix<double> zeros<double>(size_t, size_t);
template Matrix<float> zeros<float>(size_t, size_t);

} // namespace ms
