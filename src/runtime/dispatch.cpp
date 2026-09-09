// MathScript Runtime Dispatch Implementation

#include "ms/runtime/dispatch.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/runtime/thread_pool.hpp"
#include "ms/frameworks/gria/gria.hpp"

namespace ms {

namespace {

constexpr size_t kGpuMatmulThreshold = 256;

// Per-op-class GPU offload thresholds. These are dispatch POLICY defaults, not
// measurements: DenseMatmul keeps the threshold this runtime has always used, and
// the others are set from the arithmetic intensity of the op class (an FFT or a
// sparse product moves far more memory per flop than a dense GEMM, so it has to be
// bigger before an offload pays for the transfer).
size_t threshold_for(OpClass op) {
    switch (op) {
    case OpClass::DenseMatmul:
        return kGpuMatmulThreshold;
    case OpClass::FFT:
        return 4096;
    case OpClass::SparseMatmul:
        return 65536;
    case OpClass::SpecialFunction:
        return 65536;
    case OpClass::General:
        break;
    }
    return kGpuMatmulThreshold;
}

} // namespace

size_t gpu_offload_threshold(OpClass op) {
    return threshold_for(op);
}

const char* gria_hint_key(OpClass op) {
    switch (op) {
    case OpClass::DenseMatmul:
        return "matmul";
    case OpClass::FFT:
        return "fft";
    case OpClass::SparseMatmul:
        return "spmm";
    case OpClass::SpecialFunction:
        return "special";
    case OpClass::General:
        break;
    }
    return "general";
}

DispatchDecision decide(size_t n, ExecPolicy policy) {
    return decide(n, OpClass::DenseMatmul, policy, cached_topology());
}

DispatchDecision decide(size_t n, ExecPolicy policy, const SystemTopology& topo) {
    return decide(n, OpClass::DenseMatmul, policy, topo);
}

DispatchDecision decide(size_t n, OpClass op, ExecPolicy policy) {
    return decide(n, op, policy, cached_topology());
}

DispatchDecision decide(size_t n, OpClass op, ExecPolicy policy, const SystemTopology& topo) {
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

    const size_t threshold = threshold_for(op);
    const double hint_alpha = gria::dispatch_hint_alpha(gria_hint_key(op));
    if (hint_alpha >= 0.0 &&
        gria::classify(hint_alpha) == gria::ComputeClass::Irreversible &&
        has_cuda() && n >= threshold) {
        d.policy = ExecPolicy::GPU;
        d.backend = Backend::CUDA;
        d.n_threads = 0;
        d.cuda_device = topo.total_gpus > 0 ? 0 : -1;
        return d;
    }

    if (has_cuda() && n >= threshold * threshold) {
        d.policy = ExecPolicy::GPU;
        d.backend = Backend::CUDA;
        d.n_threads = 0;
    } else {
        d.policy = ExecPolicy::CPU;
        d.backend = Backend::CPU;
    }
    return d;
}

bool execute(const DispatchDecision& decision) {
    if (decision.backend == Backend::CPU && decision.n_threads > 0) {
        ThreadPool::instance().initialize(decision.n_threads);
        return true;
    }
    return false;
}

ExecPolicy get_policy_from_error() {
    return ExecPolicy::CPU;
}

} // namespace ms
