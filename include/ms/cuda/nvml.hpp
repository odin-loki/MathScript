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
