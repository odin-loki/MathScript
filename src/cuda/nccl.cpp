#include "ms/cuda/nccl.hpp"
#include "ms/cuda/buffer.hpp"

namespace ms::cuda {

bool nccl_available() {
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    return available();
#else
    return false;
#endif
}

int nccl_device_count() {
    return nccl_available() ? device_count() : 0;
}

size_t nccl_comm_size() {
    return nccl_available() ? static_cast<size_t>(device_count()) : 1;
}

} // namespace ms::cuda
