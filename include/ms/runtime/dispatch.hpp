// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
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

// Class of operation being dispatched. The GPU offload thresholds and the GRIA
// dispatch hint are keyed on this: a dense GEMM is worth offloading far earlier
// than a sparse SpMV or an elementwise special-function map.
enum class OpClass : uint8_t {
    General = 0,
    DenseMatmul,
    SparseMatmul,
    FFT,
    SpecialFunction
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

// Decision making.
//
// The overloads without an OpClass dispatch as OpClass::DenseMatmul, which is the
// policy this runtime has always applied to every caller.
DispatchDecision decide(size_t n, ExecPolicy policy);
DispatchDecision decide(size_t n, ExecPolicy policy, const SystemTopology& topo);
DispatchDecision decide(size_t n, OpClass op, ExecPolicy policy);
DispatchDecision decide(size_t n, OpClass op, ExecPolicy policy, const SystemTopology& topo);

// Problem size at or above which an op of this class is considered for GPU
// offload, and the GRIA hint registry key it consults. Exposed so callers and
// tests can see the policy rather than infer it.
size_t gpu_offload_threshold(OpClass op);
const char* gria_hint_key(OpClass op);

// Apply a dispatch decision.
//
// This binds the part of the decision the runtime can bind today: a CPU decision
// sizes ThreadPool::instance() to decision.n_threads. decision.cuda_device and
// decision.cuda_stream are NOT bound here because ms::cuda exposes no public
// device- or stream-selection entry point; the CUDA backends select their device
// when they allocate. Returns true when something was applied, false when the
// decision carried nothing this function can act on.
bool execute(const DispatchDecision& decision);

// Get policy from error
ExecPolicy get_policy_from_error();

} // namespace ms
