#include "ms/cuda/nvml.hpp"
#include "ms/cuda/buffer.hpp"

#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace ms::cuda {

bool nvml_available() {
    return false;
}

DeviceStats device_stats(int device) {
    DeviceStats stats;
    stats.name = device_name(device);
#if defined(MS_HAS_CUDA) && MS_HAS_CUDA
    if (available() && device >= 0 && device < device_count()) {
        cudaSetDevice(device);
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            stats.memory_total_bytes = total_bytes;
            stats.memory_used_bytes = total_bytes - free_bytes;
        }
    }
#endif
    return stats;
}

size_t device_memory_free(int device) {
    const DeviceStats stats = device_stats(device);
    return stats.memory_total_bytes >= stats.memory_used_bytes
        ? stats.memory_total_bytes - stats.memory_used_bytes
        : 0;
}

} // namespace ms::cuda
