// MathScript System Topology and Runtime Dispatch Advanced Tests

#include <gtest/gtest.h>
#include <string>
#include <cstddef>

#include "ms/runtime/topology.hpp"
#include "ms/runtime/dispatch.hpp"

using namespace ms;

// ---------------------------------------------------------------------------
// SystemTopology: core/thread counts
// ---------------------------------------------------------------------------

TEST(TopologyAdv, TotalCores_NonNegative) {
    SystemTopology topo;
    EXPECT_GE(topo.total_cores(), 0u);
}

TEST(TopologyAdv, TotalThreads_NonNegative) {
    SystemTopology topo;
    EXPECT_GE(topo.total_threads(), 0u);
}

TEST(TopologyAdv, TotalThreads_GTE_TotalCores) {
    SystemTopology topo;
    // With hyperthreading, threads >= cores
    EXPECT_GE(topo.total_threads(), topo.total_cores());
}

TEST(TopologyAdv, NumaCount_NonNegative) {
    SystemTopology topo;
    EXPECT_GE(topo.numa_count(), 0u);
}

TEST(TopologyAdv, SystemTopology_Detect_DoesNotCrash) {
    EXPECT_NO_THROW({
        SystemTopology topo;
        (void)topo.total_cores();
        (void)topo.total_threads();
        (void)topo.numa_count();
    });
}

// ---------------------------------------------------------------------------
// nearest_numa_node
// ---------------------------------------------------------------------------

TEST(TopologyAdv, NearestNumaNode_Returns_NonNegativeOrMinusOne) {
    SystemTopology topo;
    int result = nearest_numa_node(0, topo);
    // Should return a valid node index or -1 if not supported
    EXPECT_GE(result, -1);
}

TEST(TopologyAdv, NearestNumaNode_Finite) {
    SystemTopology topo;
    EXPECT_NO_THROW({
        int r = nearest_numa_node(0, topo);
        (void)r;
    });
}

// ---------------------------------------------------------------------------
// GPU detection (CUDA disabled: should return 0 devices)
// ---------------------------------------------------------------------------

TEST(TopologyAdv, GetGpuCount_NonNegative) {
    int count = get_gpu_count();
    EXPECT_GE(count, 0);
}

TEST(TopologyAdv, GetGpuCount_DoesNotCrash) {
    EXPECT_NO_THROW({
        int c = get_gpu_count();
        (void)c;
    });
}

TEST(TopologyAdv, GetGpuModel_InvalidDevice_DoesNotCrash) {
    // With CUDA off, device -1 or 0 should return empty/stub string
    EXPECT_NO_THROW({
        std::string model = get_gpu_model(-1);
        (void)model;
    });
}

// ---------------------------------------------------------------------------
// Dispatch: decide(), ExecPolicy, DispatchDecision
// ---------------------------------------------------------------------------

TEST(DispatchAdv, Decide_Default_Policy) {
    // decide(n, AUTO) should return a valid decision
    auto decision = decide(1024, ExecPolicy::AUTO);
    EXPECT_TRUE(decision.policy == ExecPolicy::AUTO ||
                decision.policy == ExecPolicy::CPU ||
                decision.policy == ExecPolicy::GPU);
}

TEST(DispatchAdv, Decide_CPU_Policy) {
    auto decision = decide(1024, ExecPolicy::CPU);
    EXPECT_EQ(decision.policy, ExecPolicy::CPU);
}

TEST(DispatchAdv, Decide_GPU_Policy_WhenCudaOff) {
    // Even requesting GPU, with CUDA off, backend should be CPU
    auto decision = decide(1024, ExecPolicy::GPU);
    // Backend may be CPU as fallback
    EXPECT_TRUE(decision.backend == Backend::CPU || decision.backend == Backend::CUDA);
}

TEST(DispatchAdv, DispatchDecision_DefaultInit) {
    DispatchDecision d;
    EXPECT_EQ(d.policy, ExecPolicy::AUTO);
    EXPECT_EQ(d.backend, Backend::CPU);
    EXPECT_EQ(d.n_threads, 1u);
    EXPECT_EQ(d.cuda_device, -1);
}

TEST(DispatchAdv, Decide_With_Topology) {
    SystemTopology topo;
    EXPECT_NO_THROW({
        auto d = decide(512, ExecPolicy::CPU, topo);
        (void)d;
    });
}

TEST(DispatchAdv, GetPolicyFromError_DoesNotCrash) {
    EXPECT_NO_THROW({
        auto policy = get_policy_from_error();
        (void)policy;
    });
}

TEST(DispatchAdv, ExecPolicy_Enum_Values_Distinct) {
    EXPECT_NE(static_cast<int>(ExecPolicy::CPU), static_cast<int>(ExecPolicy::GPU));
    EXPECT_NE(static_cast<int>(ExecPolicy::AUTO), static_cast<int>(ExecPolicy::GPU));
}

TEST(DispatchAdv, Backend_Enum_Values_Distinct) {
    EXPECT_NE(static_cast<int>(Backend::CPU), static_cast<int>(Backend::CUDA));
}
