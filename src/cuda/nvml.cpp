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

} // namespace ms::cuda
