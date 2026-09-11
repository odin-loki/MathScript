// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// SPDX-FileComment: links the NVIDIA CUDA libraries; see LICENSE.exceptions
#pragma once

#include <cstddef>
#include <string>

namespace ms::cuda {

struct DeviceStats {
    double utilization_pct = 0.0;
    size_t memory_used_bytes = 0;
    size_t memory_total_bytes = 0;
    std::string name;
};

bool nvml_available();
DeviceStats device_stats(int device = 0);
size_t device_memory_free(int device = 0);

} // namespace ms::cuda
