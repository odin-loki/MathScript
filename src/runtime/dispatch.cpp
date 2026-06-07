// MathScript Runtime Dispatch Implementation

#include "ms/runtime/dispatch.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/frameworks/gria/gria.hpp"

namespace ms {

namespace {

constexpr size_t kGpuMatmulThreshold = 256;

} // namespace

DispatchDecision decide(size_t n, ExecPolicy policy) {
    return decide(n, policy, detect_topology());
}

DispatchDecision decide(size_t n, ExecPolicy policy, const SystemTopology& topo) {
    DispatchDecision d;
    d.n_threads = static_cast<size_t>(topo.total_threads());
    d.cuda_device = topo.total_gpus > 0 ? 0 : -1;

    if (policy == ExecPolicy::GPU) {
        d.policy = ExecPolicy::GPU;
        d.backend = has_cuda() ? Backend::CUDA : Backend::CPU;
        d.n_threads = 0;
        return d;
    }

    if (policy == ExecPolicy::CPU) {
        d.policy = ExecPolicy::CPU;
        d.backend = Backend::CPU;
        return d;
    }

    const double matmul_alpha = gria::dispatch_hint_alpha("matmul");
    if (matmul_alpha >= 0.0 &&
        gria::classify(matmul_alpha) == gria::ComputeClass::Irreversible &&
        has_cuda() && n >= kGpuMatmulThreshold) {
        d.policy = ExecPolicy::GPU;
        d.backend = Backend::CUDA;
        d.n_threads = 0;
        d.cuda_device = topo.total_gpus > 0 ? 0 : -1;
        return d;
    }

    if (has_cuda() && n >= kGpuMatmulThreshold * kGpuMatmulThreshold) {
        d.policy = ExecPolicy::GPU;
        d.backend = Backend::CUDA;
        d.n_threads = 0;
    } else {
        d.policy = ExecPolicy::CPU;
        d.backend = Backend::CPU;
    }
    return d;
}

void execute(const DispatchDecision&) {
}

ExecPolicy get_policy_from_error() {
    return ExecPolicy::CPU;
}

} // namespace ms
