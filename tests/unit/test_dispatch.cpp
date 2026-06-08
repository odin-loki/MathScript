#include <gtest/gtest.h>
#include "ms/runtime/dispatch.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/frameworks/gria/gria.hpp"

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
    if (gpus > 0) {
        topo.gpu_devices.push_back("synthetic-gpu");
    }
    return topo;
}

} // namespace

TEST(DispatchTest, auto_cpu_when_no_gpu) {
    const auto decision = decide(512, ExecPolicy::AUTO);
    if (!has_cuda()) {
        EXPECT_EQ(decision.backend, Backend::CPU);
    }
}

TEST(DispatchTest, force_gpu_falls_back_without_hardware) {
    const auto decision = decide(64, ExecPolicy::GPU);
    if (!has_cuda()) {
        EXPECT_EQ(decision.backend, Backend::CPU);
    } else {
        EXPECT_EQ(decision.backend, Backend::CUDA);
    }
}

TEST(DispatchTest, force_cpu) {
    const auto decision = decide(4096, ExecPolicy::CPU);
    EXPECT_EQ(decision.backend, Backend::CPU);
    EXPECT_EQ(decision.policy, ExecPolicy::CPU);
}

TEST(DispatchTest, large_auto_may_select_gpu) {
    const auto decision = decide(1024, ExecPolicy::AUTO);
    if (has_cuda()) {
        EXPECT_EQ(decision.backend, Backend::CUDA);
    }
}

TEST(DispatchTest, small_auto_stays_cpu) {
    const auto decision = decide(32, ExecPolicy::AUTO);
    EXPECT_EQ(decision.backend, Backend::CPU);
}

TEST(DispatchTest, execute_and_error_policy_smoke) {
    const auto decision = decide(64, ExecPolicy::CPU);
    execute(decision);
    EXPECT_EQ(get_policy_from_error(), ExecPolicy::CPU);
}

TEST(DispatchTest, explicit_topology_sets_thread_count) {
    const auto topo = synthetic_topology(12, 0);
    const auto decision = decide(64, ExecPolicy::CPU, topo);
    EXPECT_EQ(decision.backend, Backend::CPU);
    EXPECT_EQ(decision.n_threads, 12u);
    EXPECT_EQ(decision.cuda_device, -1);
}

TEST(DispatchTest, explicit_topology_records_gpu_slot) {
    const auto topo = synthetic_topology(4, 2);
    const auto decision = decide(32, ExecPolicy::CPU, topo);
    EXPECT_EQ(decision.backend, Backend::CPU);
    EXPECT_EQ(decision.cuda_device, 0);
}

TEST(DispatchTest, auto_large_n_stays_cpu_without_cuda) {
    const auto topo = synthetic_topology(8, 1);
    const auto decision = decide(70000, ExecPolicy::AUTO, topo);
    if (!has_cuda()) {
        EXPECT_EQ(decision.backend, Backend::CPU);
        EXPECT_EQ(decision.policy, ExecPolicy::CPU);
        EXPECT_EQ(decision.n_threads, 8u);
    }
}

TEST(DispatchTest, auto_medium_n_below_gpu_threshold) {
    const auto decision = decide(512, ExecPolicy::AUTO);
    if (!has_cuda()) {
        EXPECT_EQ(decision.backend, Backend::CPU);
        EXPECT_EQ(decision.policy, ExecPolicy::CPU);
    }
}

TEST(DispatchTest, gria_matmul_hint_is_irreversible) {
    EXPECT_GE(gria::dispatch_hint_alpha("matmul"), 0.0);
    EXPECT_EQ(gria::classify(gria::dispatch_hint_alpha("matmul")), gria::ComputeClass::Irreversible);
}

TEST(DispatchTest, auto_gria_gpu_path_when_cuda_available) {
    const auto topo = synthetic_topology(16, 1);
    const auto decision = decide(256, ExecPolicy::AUTO, topo);
    if (has_cuda()) {
        EXPECT_EQ(decision.backend, Backend::CUDA);
        EXPECT_EQ(decision.policy, ExecPolicy::GPU);
        EXPECT_EQ(decision.n_threads, 0u);
        EXPECT_EQ(decision.cuda_device, 0);
    } else {
        EXPECT_EQ(decision.backend, Backend::CPU);
    }
}

TEST(DispatchTest, auto_fallback_cpu_between_thresholds) {
    const auto decision = decide(300, ExecPolicy::AUTO);
    if (!has_cuda()) {
        EXPECT_EQ(decision.backend, Backend::CPU);
    }
}
