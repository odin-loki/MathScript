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

TEST(LoadBalancerTest, two_arg_matches_three_arg_detected_topo) {
    const auto topo = detect_topology();
    const auto two_arg = balance(256, ExecPolicy::CPU);
    const auto three_arg = balance(256, ExecPolicy::CPU, topo);
    EXPECT_EQ(two_arg.backend, three_arg.backend);
    EXPECT_EQ(two_arg.cpu_threads, three_arg.cpu_threads);
    EXPECT_EQ(two_arg.cuda_device, three_arg.cuda_device);
}

TEST(LoadBalancerTest, two_arg_auto_large_workload) {
    const auto decision = balance(70000, ExecPolicy::AUTO);
    if (has_cuda()) {
        EXPECT_EQ(decision.backend, Backend::CUDA);
        EXPECT_GE(decision.cuda_device, 0);
    } else {
        EXPECT_EQ(decision.backend, Backend::CPU);
        EXPECT_GE(decision.cpu_threads, 1u);
    }
}

TEST(LoadBalancerTest, two_arg_gpu_policy) {
    const auto decision = balance(4096, ExecPolicy::GPU);
    if (has_cuda()) {
        EXPECT_EQ(decision.backend, Backend::CUDA);
        EXPECT_GE(decision.cuda_device, 0);
    } else {
        EXPECT_EQ(decision.backend, Backend::CPU);
    }
}

TEST(LoadBalancerTest, auto_large_with_explicit_topology) {
    const auto topo = synthetic_topology(8, 2);
    const auto decision = balance(70000, ExecPolicy::AUTO, topo);
    EXPECT_EQ(decision.cpu_threads, 8u);
    if (has_cuda()) {
        EXPECT_EQ(decision.backend, Backend::CUDA);
        EXPECT_GE(decision.cuda_device, 0);
        EXPECT_LT(decision.cuda_device, 2);
    } else {
        EXPECT_EQ(decision.backend, Backend::CPU);
    }
}

TEST(LoadBalancerTest, empty_topology_zero_threads) {
    SystemTopology topo;
    topo.total_gpus = 0;
    const auto decision = balance(64, ExecPolicy::CPU, topo);
    EXPECT_EQ(decision.backend, Backend::CPU);
    EXPECT_EQ(decision.cpu_threads, 0u);
    EXPECT_EQ(decision.cuda_device, -1);
}

TEST(LoadBalancerTest, zero_thread_cpu_threads_empty) {
    SystemTopology topo;
    topo.cpu_cores = {{0, 1}};
    topo.total_gpus = 0;
    EXPECT_TRUE(topo.cpu_threads.empty());
    EXPECT_EQ(topo.total_threads(), 0u);
    const auto decision = balance(128, ExecPolicy::AUTO, topo);
    EXPECT_EQ(decision.cpu_threads, 0u);
    EXPECT_EQ(decision.backend, Backend::CPU);
}

TEST(LoadBalancerTest, gpu_policy_two_gpus_picks_device) {
    const auto topo = synthetic_topology(4, 2);
    EXPECT_GT(topo.total_gpus, 1);
    const auto decision = balance(1024, ExecPolicy::GPU, topo);
    if (!has_cuda()) {
        GTEST_SKIP() << "CUDA unavailable; multi-GPU picker not under test";
    }
    EXPECT_EQ(decision.backend, Backend::CUDA);
    EXPECT_GE(decision.cuda_device, 0);
    EXPECT_LT(decision.cuda_device, topo.total_gpus);
}
