#include "ms/cuda/nvml.hpp"
#include "ms/cuda/buffer.hpp"

namespace ms::cuda {

bool nvml_available() {
    return false;
}

DeviceStats device_stats(int device) {
    DeviceStats stats;
    stats.name = device_name(device);
    if (available()) {
        stats.memory_total_bytes = 8ULL * 1024 * 1024 * 1024;
    }
    return stats;
}

size_t device_memory_free(int device) {
    const DeviceStats stats = device_stats(device);
    return stats.memory_total_bytes >= stats.memory_used_bytes
        ? stats.memory_total_bytes - stats.memory_used_bytes
        : 0;
}

} // namespace ms::cuda
