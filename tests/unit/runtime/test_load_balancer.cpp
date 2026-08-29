#include <gtest/gtest.h>
#include "ms/cuda/nccl.hpp"
#include "ms/runtime/load_balancer.hpp"
#include "ms/runtime/topology.hpp"

using namespace ms;

namespace {

SystemTopology synthetic_topology(size_t threads, int gpus) {
    SystemTopology topo;
    topo.cpu_cores = {{}};
    topo.cpu_threads = {{}};
    for (size_t i = 0; i < threads; ++i) {
        topo.cpu_cores[0].push_back(i);
        topo.cpu_threads[0].push_back(i);
    }
    topo.numa_nodes = {0};
    topo.total_gpus = gpus;
    for (int g = 0; g < gpus; ++g) {
        topo.gpu_devices.push_back("synthetic-gpu");
    }
    return topo;
}

} // namespace

TEST(LoadBalancerTest, auto_picks_cpu_for_small_workload) {
    const auto decision = balance(32, ExecPolicy::AUTO);
    EXPECT_EQ(decision.backend, Backend::CPU);
    EXPECT_GE(decision.cpu_threads, 1u);
}

TEST(LoadBalancerTest, gpu_policy_when_available) {
    const auto topo = detect_topology();
    const auto decision = balance(4096, ExecPolicy::GPU, topo);
    if (has_cuda()) {
        EXPECT_EQ(decision.backend, Backend::CUDA);
        EXPECT_GE(decision.cuda_device, 0);
    } else {
        EXPECT_EQ(decision.backend, Backend::CPU);
    }
}

TEST(LoadBalancerTest, nccl_stub_comm_size) {
    EXPECT_GE(cuda::nccl_comm_size(), 1u);
}

TEST(LoadBalancerTest, force_cpu_policy) {
    const auto decision = balance(8192, ExecPolicy::CPU);
    EXPECT_EQ(decision.backend, Backend::CPU);
    EXPECT_GE(decision.cpu_threads, 1u);
}

TEST(LoadBalancerTest, explicit_topology_sets_threads) {
    const auto topo = synthetic_topology(10, 0);
    const auto decision = balance(128, ExecPolicy::CPU, topo);
    EXPECT_EQ(decision.backend, Backend::CPU);
    EXPECT_EQ(decision.cpu_threads, 10u);
}

TEST(LoadBalancerTest, auto_large_without_cuda_stays_cpu) {
    const auto topo = synthetic_topology(6, 2);
    const auto decision = balance(70000, ExecPolicy::AUTO, topo);
    if (!has_cuda()) {
        EXPECT_EQ(decision.backend, Backend::CPU);
        EXPECT_EQ(decision.cpu_threads, 6u);
    }
}

TEST(LoadBalancerTest, gpu_policy_records_device) {
    const auto topo = synthetic_topology(4, 1);
    const auto decision = balance(1024, ExecPolicy::GPU, topo);
    if (has_cuda()) {
        EXPECT_EQ(decision.backend, Backend::CUDA);
        EXPECT_GE(decision.cuda_device, 0);
    } else {
        EXPECT_EQ(decision.backend, Backend::CPU);
    }
}
