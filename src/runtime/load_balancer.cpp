#include "ms/runtime/load_balancer.hpp"
#include "ms/cuda/nvml.hpp"

namespace ms {

LoadBalanceDecision balance(size_t workload, ExecPolicy policy) {
    return balance(workload, policy, detect_topology());
}

LoadBalanceDecision balance(
    size_t workload,
    ExecPolicy policy,
    const SystemTopology& topo) {
    LoadBalanceDecision decision;
    decision.cpu_threads = static_cast<size_t>(topo.total_threads());

    const auto dispatch = decide(workload, policy, topo);
    decision.backend = dispatch.backend;
    decision.cuda_device = dispatch.cuda_device;

    if (decision.backend == Backend::CUDA && topo.total_gpus > 1) {
        int best_device = 0;
        size_t best_free = 0;
        for (int device = 0; device < topo.total_gpus; ++device) {
            const auto stats = cuda::device_stats(device);
            const size_t free_bytes = stats.memory_total_bytes - stats.memory_used_bytes;
            if (free_bytes > best_free) {
                best_free = free_bytes;
                best_device = device;
            }
        }
        decision.cuda_device = best_device;
    }

    return decision;
}

} // namespace ms
