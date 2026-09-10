// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// Regression tests for the audited ms::detect_topology / ms::nearest_numa_node /
// ms::execute defects.
//
// Before the fix:
//   * detect_topology() never assigned cpu_sockets (it stayed empty) and filled
//     cpu_cores and cpu_threads with the same hardware_concurrency() list, so
//     total_cores() == total_threads() on every machine and SMT was invisible.
//   * nearest_numa_node(core_id, topo) ignored both arguments and returned 0.
//   * execute(decision) had an empty body.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "ms/runtime/dispatch.hpp"
#include "ms/runtime/thread_pool.hpp"
#include "ms/runtime/topology.hpp"

using ms::SystemTopology;

namespace {

#if defined(__linux__)
// Independently re-derive the socket set straight from sysfs so the assertion is
// not just detect_topology() agreeing with itself.
std::set<std::string> sysfs_package_ids(const std::vector<std::size_t>& cpus) {
    std::set<std::string> ids;
    for (std::size_t cpu : cpus) {
        std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                         "/topology/physical_package_id");
        std::string line;
        if (in.is_open() && std::getline(in, line) && !line.empty()) {
            ids.insert(line);
        }
    }
    return ids;
}

std::vector<std::size_t> sysfs_online_cpus() {
    std::ifstream in("/sys/devices/system/cpu/online");
    std::string spec;
    std::vector<std::size_t> cpus;
    if (!in.is_open() || !std::getline(in, spec)) {
        return cpus;
    }
    std::size_t pos = 0;
    while (pos <= spec.size()) {
        const std::size_t comma = spec.find(',', pos);
        const std::string token =
            spec.substr(pos, (comma == std::string::npos) ? std::string::npos : comma - pos);
        if (!token.empty()) {
            const std::size_t dash = token.find('-');
            if (dash == std::string::npos) {
                cpus.push_back(static_cast<std::size_t>(std::stoul(token)));
            } else {
                const auto lo = static_cast<std::size_t>(std::stoul(token.substr(0, dash)));
                const auto hi = static_cast<std::size_t>(std::stoul(token.substr(dash + 1)));
                for (std::size_t v = lo; v <= hi; ++v) {
                    cpus.push_back(v);
                }
            }
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return cpus;
}
#endif

} // namespace

TEST(TopologyContract, SocketsCoresAndThreadsAreConsistent) {
    const SystemTopology topo = ms::detect_topology();

    // cpu_sockets used to be left empty by every code path in the tree.
    EXPECT_FALSE(topo.cpu_sockets.empty());
    EXPECT_EQ(topo.cpu_sockets.size(), topo.cpu_cores.size());
    EXPECT_EQ(topo.cpu_sockets.size(), topo.cpu_threads.size());

    EXPECT_GT(topo.total_threads(), 0u);
    EXPECT_GT(topo.total_cores(), 0u);
    EXPECT_LE(topo.total_cores(), topo.total_threads());

    // Every listed physical core is also a listed logical CPU on the same socket.
    for (std::size_t s = 0; s < topo.cpu_sockets.size(); ++s) {
        EXPECT_LE(topo.cpu_cores[s].size(), topo.cpu_threads[s].size());
        for (std::size_t core : topo.cpu_cores[s]) {
            EXPECT_NE(std::find(topo.cpu_threads[s].begin(), topo.cpu_threads[s].end(), core),
                      topo.cpu_threads[s].end());
        }
    }

    EXPECT_FALSE(topo.numa_nodes.empty());
    EXPECT_GE(topo.numa_count(), 1u);
}

#if defined(__linux__)
TEST(TopologyContract, LinuxDetectionMatchesSysfsDirectly) {
    const std::vector<std::size_t> online = sysfs_online_cpus();
    if (online.empty()) {
        GTEST_SKIP() << "sysfs CPU topology not available";
    }
    const SystemTopology topo = ms::detect_topology();
    ASSERT_TRUE(topo.detected) << "sysfs is present, so detection must not fall back";

    EXPECT_EQ(topo.total_threads(), online.size());

    const std::set<std::string> packages = sysfs_package_ids(online);
    ASSERT_FALSE(packages.empty());
    EXPECT_EQ(topo.cpu_sockets.size(), packages.size());
    for (const std::string& id : topo.cpu_sockets) {
        EXPECT_TRUE(packages.count(id) == 1) << "unexpected socket id " << id;
    }

    // Every online logical CPU maps to a node that the topology also lists.
    for (std::size_t cpu : online) {
        const int node = ms::nearest_numa_node(cpu, topo);
        EXPECT_NE(std::find(topo.numa_nodes.begin(), topo.numa_nodes.end(), node),
                  topo.numa_nodes.end())
            << "cpu " << cpu << " mapped to unknown node " << node;
    }
}
#endif

// Machine-independent: the old body returned 0 for every argument.
TEST(TopologyContract, NearestNumaNodeReadsTheCpuToNodeMap) {
    SystemTopology topo;
    topo.cpu_sockets = {"0", "1"};
    topo.cpu_cores = {{0, 1}, {2, 3}};
    topo.cpu_threads = {{0, 1}, {2, 3}};
    topo.numa_nodes = {1, 2};
    topo.cpu_numa_node = {1, 1, 2, 2};
    topo.detected = true;
    topo.total_gpus = 0;

    EXPECT_EQ(ms::nearest_numa_node(0, topo), 1);
    EXPECT_EQ(ms::nearest_numa_node(1, topo), 1);
    EXPECT_EQ(ms::nearest_numa_node(2, topo), 2);
    EXPECT_EQ(ms::nearest_numa_node(3, topo), 2);

    // Out-of-range and unknown ids fall back to the first known node, not to 0.
    EXPECT_EQ(ms::nearest_numa_node(99, topo), 1);
    topo.cpu_numa_node[1] = -1;
    EXPECT_EQ(ms::nearest_numa_node(1, topo), 1);
}

TEST(TopologyContract, NearestNumaNodeOnAnEmptyTopologyIsZero) {
    SystemTopology topo;
    topo.total_gpus = 0;
    EXPECT_EQ(ms::nearest_numa_node(0, topo), 0);
}

// ---------------------------------------------------------------------------
// ms::execute  (audit finding 7)
// ---------------------------------------------------------------------------

TEST(DispatchExecuteContract, CpuDecisionSizesTheThreadPool) {
    ms::DispatchDecision decision;
    decision.policy = ms::ExecPolicy::CPU;
    decision.backend = ms::Backend::CPU;
    decision.n_threads = 3;

    EXPECT_TRUE(ms::execute(decision));

    auto& pool = ms::ThreadPool::instance();
    const std::size_t cap = pool.config().max_workers;
    EXPECT_EQ(pool.size(), std::min<std::size_t>(3u, cap));
    EXPECT_GT(pool.size(), 0u);
}

TEST(DispatchExecuteContract, DecisionWithNothingToBindReportsFalse) {
    ms::DispatchDecision decision;
    decision.policy = ms::ExecPolicy::CPU;
    decision.backend = ms::Backend::CPU;
    decision.n_threads = 0;
    EXPECT_FALSE(ms::execute(decision));
}

// ---------------------------------------------------------------------------
// OpClass-aware dispatch (audit finding 8: the orphan header advertised this API
// with no definition anywhere; it is now part of the real header).
// ---------------------------------------------------------------------------

TEST(DispatchOpClassContract, ThresholdsAndHintKeysAreDistinctPerOpClass) {
    EXPECT_STREQ(ms::gria_hint_key(ms::OpClass::DenseMatmul), "matmul");
    EXPECT_STREQ(ms::gria_hint_key(ms::OpClass::FFT), "fft");
    EXPECT_STREQ(ms::gria_hint_key(ms::OpClass::SparseMatmul), "spmm");
    EXPECT_STREQ(ms::gria_hint_key(ms::OpClass::SpecialFunction), "special");
    EXPECT_STREQ(ms::gria_hint_key(ms::OpClass::General), "general");

    EXPECT_EQ(ms::gpu_offload_threshold(ms::OpClass::DenseMatmul), 256u);
    EXPECT_GT(ms::gpu_offload_threshold(ms::OpClass::FFT),
              ms::gpu_offload_threshold(ms::OpClass::DenseMatmul));
    EXPECT_GT(ms::gpu_offload_threshold(ms::OpClass::SparseMatmul),
              ms::gpu_offload_threshold(ms::OpClass::FFT));
}

TEST(DispatchOpClassContract, LegacyOverloadsStillDispatchAsDenseMatmul) {
    const SystemTopology topo = ms::detect_topology();
    for (std::size_t n : {std::size_t{1}, std::size_t{300}, std::size_t{100000}}) {
        const ms::DispatchDecision legacy = ms::decide(n, ms::ExecPolicy::AUTO, topo);
        const ms::DispatchDecision explicit_op =
            ms::decide(n, ms::OpClass::DenseMatmul, ms::ExecPolicy::AUTO, topo);
        EXPECT_EQ(legacy.backend, explicit_op.backend) << "n=" << n;
        EXPECT_EQ(legacy.policy, explicit_op.policy) << "n=" << n;
        EXPECT_EQ(legacy.n_threads, explicit_op.n_threads) << "n=" << n;
    }
}

TEST(DispatchOpClassContract, CpuAndGpuPoliciesAreHonouredForEveryOpClass) {
    const SystemTopology topo = ms::detect_topology();
    const ms::OpClass classes[] = {ms::OpClass::General, ms::OpClass::DenseMatmul,
                                   ms::OpClass::SparseMatmul, ms::OpClass::FFT,
                                   ms::OpClass::SpecialFunction};
    for (ms::OpClass op : classes) {
        const ms::DispatchDecision cpu = ms::decide(1024, op, ms::ExecPolicy::CPU, topo);
        EXPECT_EQ(cpu.backend, ms::Backend::CPU);
        EXPECT_EQ(cpu.policy, ms::ExecPolicy::CPU);
        EXPECT_GT(cpu.n_threads, 0u);
    }
}
