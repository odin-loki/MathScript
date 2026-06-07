#include <gtest/gtest.h>
#include "ms/cuda/nccl.hpp"
#include "ms/runtime/load_balancer.hpp"
#include "ms/runtime/topology.hpp"

using namespace ms;

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
