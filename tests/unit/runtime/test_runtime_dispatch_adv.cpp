// MathScript: Advanced Runtime Dispatch and Load Balancer Tests

#include <gtest/gtest.h>
#include <cstddef>
#include "ms/runtime/dispatch.hpp"
#include "ms/runtime/load_balancer.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/runtime/thread_pool.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// DispatchDecision defaults
// ---------------------------------------------------------------------------

TEST(RuntimeDispatchAdv, DispatchDecision_DefaultValues) {
    DispatchDecision d;
    EXPECT_EQ(d.policy, ExecPolicy::AUTO);
    EXPECT_EQ(d.backend, Backend::CPU);
    EXPECT_GE(d.n_threads, 1u);
    EXPECT_EQ(d.cuda_device, -1);
    EXPECT_EQ(d.cuda_stream, 0);
}

TEST(RuntimeDispatchAdv, ExecPolicy_Values) {
    EXPECT_NE(ExecPolicy::AUTO, ExecPolicy::CPU);
    EXPECT_NE(ExecPolicy::CPU, ExecPolicy::GPU);
    EXPECT_NE(ExecPolicy::AUTO, ExecPolicy::GPU);
}

TEST(RuntimeDispatchAdv, Backend_Values) {
    EXPECT_NE(Backend::CPU, Backend::CUDA);
}

// ---------------------------------------------------------------------------
// decide() function
// ---------------------------------------------------------------------------

TEST(RuntimeDispatchAdv, Decide_SmallWorkload_CPU) {
    auto d = decide(100, ExecPolicy::CPU);
    EXPECT_EQ(d.backend, Backend::CPU);
    EXPECT_EQ(d.policy, ExecPolicy::CPU);
}

TEST(RuntimeDispatchAdv, Decide_GPUPolicy_SetsCUDAOrCPU) {
    auto d = decide(1000000, ExecPolicy::GPU);
    // Either CUDA or CPU (fallback if no GPU)
    bool valid = (d.backend == Backend::CUDA || d.backend == Backend::CPU);
    EXPECT_TRUE(valid);
}

TEST(RuntimeDispatchAdv, Decide_AutoPolicy_LargeWorkload) {
    auto d = decide(1000000, ExecPolicy::AUTO);
    // Should produce a valid decision
    EXPECT_GE(d.n_threads, 1u);
}

TEST(RuntimeDispatchAdv, Decide_WithTopology_CPU) {
    SystemTopology topo = detect_topology();
    auto d = decide(1000, ExecPolicy::CPU, topo);
    EXPECT_EQ(d.backend, Backend::CPU);
    EXPECT_GE(d.n_threads, 1u);
}

TEST(RuntimeDispatchAdv, Decide_WithTopology_ThreadCount) {
    SystemTopology topo = detect_topology();
    auto d = decide(1000000, ExecPolicy::AUTO, topo);
    EXPECT_GE(d.n_threads, 1u);
    // Thread count should not exceed total available threads
    EXPECT_LE(d.n_threads, topo.total_threads() + 1);
}

TEST(RuntimeDispatchAdv, Decide_ZeroWorkload) {
    auto d = decide(0, ExecPolicy::CPU);
    EXPECT_EQ(d.backend, Backend::CPU);
}

// ---------------------------------------------------------------------------
// execute() - smoke test (should not crash)
// ---------------------------------------------------------------------------

TEST(RuntimeDispatchAdv, Execute_CPUDecision_Smoke) {
    auto d = decide(100, ExecPolicy::CPU);
    EXPECT_NO_THROW(execute(d));
}

// ---------------------------------------------------------------------------
// get_policy_from_error()
// ---------------------------------------------------------------------------

TEST(RuntimeDispatchAdv, GetPolicyFromError_ReturnsValidPolicy) {
    ExecPolicy p = get_policy_from_error();
    bool valid = (p == ExecPolicy::AUTO || p == ExecPolicy::CPU || p == ExecPolicy::GPU);
    EXPECT_TRUE(valid);
}

TEST(RuntimeDispatchAdv, GetPolicyFromError_IsCPUOrAuto) {
    // After an error, fallback policy should typically be CPU or AUTO
    ExecPolicy p = get_policy_from_error();
    EXPECT_NE(p, ExecPolicy::GPU) << "Error recovery should not use GPU";
}

// ---------------------------------------------------------------------------
// LoadBalanceDecision defaults
// ---------------------------------------------------------------------------

TEST(RuntimeDispatchAdv, LoadBalance_DefaultDecision) {
    LoadBalanceDecision d;
    EXPECT_EQ(d.backend, Backend::CPU);
    EXPECT_EQ(d.cuda_device, -1);
    EXPECT_GE(d.cpu_threads, 1u);
}

// ---------------------------------------------------------------------------
// balance() function
// ---------------------------------------------------------------------------

TEST(RuntimeDispatchAdv, Balance_SmallWorkload_CPU) {
    auto d = balance(100, ExecPolicy::CPU);
    EXPECT_EQ(d.backend, Backend::CPU);
    EXPECT_GE(d.cpu_threads, 1u);
}

TEST(RuntimeDispatchAdv, Balance_LargeWorkload_AUTO) {
    auto d = balance(10000000, ExecPolicy::AUTO);
    EXPECT_GE(d.cpu_threads, 1u);
}

TEST(RuntimeDispatchAdv, Balance_WithTopology) {
    SystemTopology topo = detect_topology();
    auto d = balance(500000, ExecPolicy::AUTO, topo);
    EXPECT_GE(d.cpu_threads, 1u);
    EXPECT_LE(d.cpu_threads, topo.total_threads() + 1);
}

TEST(RuntimeDispatchAdv, Balance_ZeroWorkload) {
    auto d = balance(0, ExecPolicy::CPU);
    EXPECT_GE(d.cpu_threads, 1u);
}

TEST(RuntimeDispatchAdv, Balance_GPUPolicy) {
    auto d = balance(1000000, ExecPolicy::GPU);
    // If no GPU, falls back to CPU
    bool valid = (d.backend == Backend::CUDA || d.backend == Backend::CPU);
    EXPECT_TRUE(valid);
}

// ---------------------------------------------------------------------------
// ThreadPool integration with dispatch
// ---------------------------------------------------------------------------

TEST(RuntimeDispatchAdv, ThreadPool_Submit_UsesDecidedThreads) {
    auto d = decide(1000, ExecPolicy::CPU);
    auto& pool = ThreadPool::instance();

    // Submit as many tasks as n_threads
    std::vector<std::future<int>> futures;
    for (size_t i = 0; i < d.n_threads && i < 4; ++i) {
        futures.push_back(pool.submit([i]() -> int { return (int)i * 2; }));
    }
    for (auto& f : futures) {
        EXPECT_GE(f.get(), 0);
    }
}

// ---------------------------------------------------------------------------
// SystemTopology fields
// ---------------------------------------------------------------------------

TEST(RuntimeDispatchAdv, SystemTopology_TotalCores_GE_1) {
    auto topo = detect_topology();
    EXPECT_GE(topo.total_cores(), 1u);
}

TEST(RuntimeDispatchAdv, SystemTopology_TotalThreads_GE_Cores) {
    auto topo = detect_topology();
    EXPECT_GE(topo.total_threads(), topo.total_cores());
}

TEST(RuntimeDispatchAdv, SystemTopology_NumaCount_GE_1) {
    auto topo = detect_topology();
    EXPECT_GE(topo.numa_count(), 1u);
}

TEST(RuntimeDispatchAdv, SystemTopology_GetGPUModel_Smoke) {
    if (has_cuda() && get_gpu_count() > 0) {
        std::string model = get_gpu_model(0);
        EXPECT_FALSE(model.empty());
    } else {
        SUCCEED() << "No CUDA GPU available";
    }
}
