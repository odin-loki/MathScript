// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once

#include "ms/runtime/dispatch.hpp"
#include "ms/runtime/topology.hpp"

namespace ms {

struct LoadBalanceDecision {
    Backend backend = Backend::CPU;
    int cuda_device = -1;
    size_t cpu_threads = 1;
};

LoadBalanceDecision balance(size_t workload, ExecPolicy policy = ExecPolicy::AUTO);
LoadBalanceDecision balance(
    size_t workload,
    ExecPolicy policy,
    const SystemTopology& topo);

} // namespace ms
