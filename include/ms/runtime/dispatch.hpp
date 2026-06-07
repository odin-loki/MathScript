// MathScript Runtime Dispatch Header

#pragma once

#include <cstdint>
#include "ms/runtime/topology.hpp"

namespace ms {

// Execution policy
enum class ExecPolicy : uint8_t {
    AUTO = 0,
    CPU = 1,
    GPU = 2
};

// Backend
enum class Backend : uint8_t {
    CPU,
    CUDA
};

// Dispatch decision
struct DispatchDecision {
    ExecPolicy policy;
    Backend backend;
    size_t n_threads;
    int cuda_device;
    int cuda_stream;
    
    DispatchDecision() 
        : policy(ExecPolicy::AUTO),
          backend(Backend::CPU),
          n_threads(1),
          cuda_device(-1),
          cuda_stream(0) {}
};

// Decision making
DispatchDecision decide(size_t n, ExecPolicy policy);
DispatchDecision decide(size_t n, ExecPolicy policy, const SystemTopology& topo);

// Execute
void execute(const DispatchDecision& decision);

// Get policy from error
ExecPolicy get_policy_from_error();

} // namespace ms