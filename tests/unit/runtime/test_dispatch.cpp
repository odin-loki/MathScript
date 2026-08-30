#include <gtest/gtest.h>
#include "ms/frameworks/gria/gria.hpp"
#include "ms/runtime/dispatch.hpp"
#include "ms/runtime/topology.hpp"

using namespace ms;

TEST(DispatchTest, default_construction) {
    DispatchDecision d;
    EXPECT_EQ(d.policy, ExecPolicy::AUTO);
    EXPECT_EQ(d.backend, Backend::CPU);
    EXPECT_EQ(d.n_threads, 1u);
    EXPECT_EQ(d.cuda_device, -1);
    EXPECT_EQ(d.cuda_stream, 0);
}

TEST(DispatchTest, decide_auto_small_workload_uses_cpu) {
    const auto d = decide(100, ExecPolicy::AUTO);
    // No CUDA in test environment; AUTO should resolve to CPU
    EXPECT_EQ(d.backend, Backend::CPU);
    EXPECT_GE(d.n_threads, 1u);
}

TEST(DispatchTest, decide_cpu_policy_always_cpu) {
    const auto d = decide(1000000, ExecPolicy::CPU);
    EXPECT_EQ(d.backend, Backend::CPU);
}

TEST(DispatchTest, decide_gpu_policy_fallback_when_no_cuda) {
    const auto d = decide(1000, ExecPolicy::GPU);
    // In no-CUDA build, GPU policy should not crash and backend is set
    (void)d.backend;
}

TEST(DispatchTest, decide_with_topology) {
    const SystemTopology topo = detect_topology();
    const auto d = decide(1024, ExecPolicy::AUTO, topo);
    EXPECT_GE(d.n_threads, 1u);
    EXPECT_TRUE(d.backend == Backend::CPU || d.backend == Backend::CUDA);
}

TEST(DispatchTest, execute_does_not_throw) {
    DispatchDecision d;
    d.backend = Backend::CPU;
    d.n_threads = 1;
    execute(d);
}

TEST(DispatchTest, get_policy_from_error_returns_cpu) {
    const auto policy = get_policy_from_error();
    EXPECT_EQ(policy, ExecPolicy::CPU);
}

TEST(DispatchTest, decide_large_workload_uses_more_threads) {
    const auto d_small = decide(10, ExecPolicy::CPU);
    const auto d_large = decide(10000000, ExecPolicy::CPU);
    // Large workload may use more threads
    EXPECT_GE(d_large.n_threads, d_small.n_threads);
}

TEST(DispatchTest, decide_zero_workload_no_crash) {
    {
        const auto d = decide(0, ExecPolicy::AUTO);
        (void)d.backend;
    }
}

TEST(DispatchTest, decide_two_arg_matches_three_arg_cpu) {
    const auto topo = detect_topology();
    const auto two_arg = decide(128, ExecPolicy::CPU);
    const auto three_arg = decide(128, ExecPolicy::CPU, topo);
    EXPECT_EQ(two_arg.policy, three_arg.policy);
    EXPECT_EQ(two_arg.backend, three_arg.backend);
    EXPECT_EQ(two_arg.n_threads, three_arg.n_threads);
    EXPECT_EQ(two_arg.cuda_device, three_arg.cuda_device);
}

TEST(DispatchTest, decide_cpu_policy_uses_topology_threads) {
    SystemTopology topo;
    topo.cpu_threads = {{0, 1, 2, 3, 4, 5}};
    topo.total_gpus = 0;
    const auto d = decide(1000000, ExecPolicy::CPU, topo);
    EXPECT_EQ(d.policy, ExecPolicy::CPU);
    EXPECT_EQ(d.backend, Backend::CPU);
    EXPECT_EQ(d.n_threads, 6u);
    EXPECT_EQ(d.cuda_device, -1);
}

TEST(DispatchTest, decide_gpu_policy_sets_threads_zero) {
    SystemTopology topo;
    topo.cpu_threads = {{0, 1, 2, 3}};
    topo.total_gpus = 1;
    const auto d = decide(1024, ExecPolicy::GPU, topo);
    EXPECT_EQ(d.policy, ExecPolicy::GPU);
    EXPECT_EQ(d.n_threads, 0u);
    EXPECT_EQ(d.cuda_device, 0);
    if (has_cuda()) {
        EXPECT_EQ(d.backend, Backend::CUDA);
    } else {
        EXPECT_EQ(d.backend, Backend::CPU);
    }
}

TEST(DispatchTest, decide_gpu_policy_zero_gpus_device_negative) {
    SystemTopology topo;
    topo.cpu_threads = {{0}};
    topo.total_gpus = 0;
    const auto d = decide(64, ExecPolicy::GPU, topo);
    EXPECT_EQ(d.policy, ExecPolicy::GPU);
    EXPECT_EQ(d.n_threads, 0u);
    EXPECT_EQ(d.cuda_device, -1);
    EXPECT_EQ(d.backend, has_cuda() ? Backend::CUDA : Backend::CPU);
}

TEST(DispatchTest, decide_gpu_two_arg_backend) {
    const auto d = decide(1000, ExecPolicy::GPU);
    EXPECT_EQ(d.policy, ExecPolicy::GPU);
    EXPECT_EQ(d.n_threads, 0u);
    if (has_cuda()) {
        EXPECT_EQ(d.backend, Backend::CUDA);
    } else {
        EXPECT_EQ(d.backend, Backend::CPU);
    }
}

TEST(DispatchTest, decide_auto_just_below_gpu_threshold) {
    SystemTopology topo;
    topo.cpu_threads = {{0, 1}};
    topo.total_gpus = 1;
    const auto d = decide(255, ExecPolicy::AUTO, topo);
    EXPECT_EQ(d.policy, ExecPolicy::CPU);
    EXPECT_EQ(d.backend, Backend::CPU);
    EXPECT_EQ(d.n_threads, 2u);
    EXPECT_EQ(d.cuda_device, 0);
}

TEST(DispatchTest, decide_auto_at_gpu_threshold) {
    SystemTopology topo;
    topo.cpu_threads = {{0}};
    topo.total_gpus = 1;
    const auto d = decide(256, ExecPolicy::AUTO, topo);
    if (!has_cuda()) {
        GTEST_SKIP() << "CUDA unavailable; AUTO GPU threshold branch not under test";
    }
    EXPECT_EQ(d.policy, ExecPolicy::GPU);
    EXPECT_EQ(d.backend, Backend::CUDA);
    EXPECT_EQ(d.n_threads, 0u);
    EXPECT_EQ(d.cuda_device, 0);
}

TEST(DispatchTest, decide_auto_large_workload_with_topology) {
    SystemTopology topo;
    topo.cpu_threads = {{0, 1, 2, 3}};
    topo.total_gpus = 2;
    const auto d = decide(70000, ExecPolicy::AUTO, topo);
    if (has_cuda()) {
        EXPECT_EQ(d.policy, ExecPolicy::GPU);
        EXPECT_EQ(d.backend, Backend::CUDA);
        EXPECT_EQ(d.n_threads, 0u);
        EXPECT_EQ(d.cuda_device, 0);
    } else {
        EXPECT_EQ(d.policy, ExecPolicy::CPU);
        EXPECT_EQ(d.backend, Backend::CPU);
        EXPECT_EQ(d.n_threads, 4u);
    }
}

TEST(DispatchTest, decide_auto_empty_thread_topology) {
    SystemTopology topo;
    topo.total_gpus = 0;
    EXPECT_TRUE(topo.cpu_threads.empty());
    const auto d = decide(10, ExecPolicy::AUTO, topo);
    EXPECT_EQ(d.n_threads, 0u);
    EXPECT_EQ(d.cuda_device, -1);
    EXPECT_EQ(d.policy, ExecPolicy::CPU);
    EXPECT_EQ(d.backend, Backend::CPU);
}

TEST(DispatchTest, decide_auto_square_threshold_without_irreversible_hint) {
    gria::register_dispatch_hint("matmul", 0.2);
    SystemTopology topo;
    topo.cpu_threads = {{0}};
    topo.total_gpus = 1;
    const auto d = decide(65536, ExecPolicy::AUTO, topo);
    gria::register_dispatch_hint("matmul", 0.72);
    if (!has_cuda()) {
        GTEST_SKIP() << "CUDA unavailable; square-threshold GPU branch not under test";
    }
    EXPECT_EQ(d.policy, ExecPolicy::GPU);
    EXPECT_EQ(d.backend, Backend::CUDA);
    EXPECT_EQ(d.n_threads, 0u);
}

TEST(DispatchTest, execute_gpu_decision) {
    DispatchDecision d;
    d.policy = ExecPolicy::GPU;
    d.backend = has_cuda() ? Backend::CUDA : Backend::CPU;
    d.n_threads = 0;
    d.cuda_device = 0;
    execute(d);
}

TEST(DispatchTest, execute_cuda_backend_decision) {
    DispatchDecision d;
    d.policy = ExecPolicy::GPU;
    d.backend = Backend::CUDA;
    d.n_threads = 0;
    d.cuda_device = 0;
    execute(d);
}
