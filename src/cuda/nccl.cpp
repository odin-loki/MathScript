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

double allreduce_sum(double value) {
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (!nccl_available() || nccl_comm_size() <= 1) {
        return value;
    }
    // Multi-GPU NCCL reduction is wired in a later wave; single-rank identity
    // keeps host and stub builds deterministic until comm init lands.
    return value;
#else
    return value;
#endif
}

double allreduce_max(double value) {
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (!nccl_available() || nccl_comm_size() <= 1) {
        return value;
    }
    // Multi-GPU NCCL reduction is wired in a later wave; identity until
    // MultiGPUContext and comm init land.
    return value;
#else
    return value;
#endif
}

double allreduce_min(double value) {
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (!nccl_available() || nccl_comm_size() <= 1) {
        return value;
    }
    // Multi-GPU NCCL reduction is wired in a later wave; identity until
    // MultiGPUContext and comm init land.
    return value;
#else
    return value;
#endif
}

double allreduce_prod(double value) {
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (!nccl_available() || nccl_comm_size() <= 1) {
        return value;
    }
    // Multi-GPU NCCL reduction is wired in a later wave; identity until
    // MultiGPUContext and comm init land.
    return value;
#else
    return value;
#endif
}

double allreduce_avg(double value) {
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (!nccl_available() || nccl_comm_size() <= 1) {
        return value;
    }
    const size_t size = nccl_comm_size();
    return allreduce_sum(value) / static_cast<double>(size);
#else
    return value;
#endif
}

double broadcast(double value, int /*root*/) {
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (!nccl_available() || nccl_comm_size() <= 1) {
        return value;
    }
    // Multi-GPU NCCL broadcast is wired in a later wave; identity until
    // MultiGPUContext and comm init land.
    return value;
#else
    return value;
#endif
}

double reduce(double value, int /*root*/) {
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (!nccl_available() || nccl_comm_size() <= 1) {
        return value;
    }
    // Multi-GPU NCCL reduce is wired in a later wave; identity until
    // MultiGPUContext and comm init land.
    return value;
#else
    return value;
#endif
}

double allgather(double value) {
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (!nccl_available() || nccl_comm_size() <= 1) {
        return value;
    }
    // Multi-GPU NCCL allgather is wired in a later wave; identity until
    // MultiGPUContext and comm init land.
    return value;
#else
    return value;
#endif
}

} // namespace ms::cuda
