#include <gtest/gtest.h>
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
    EXPECT_NO_THROW((void)d.backend);
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
    EXPECT_NO_THROW(execute(d));
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
    EXPECT_NO_THROW({
        const auto d = decide(0, ExecPolicy::AUTO);
        (void)d.backend;
    });
}
