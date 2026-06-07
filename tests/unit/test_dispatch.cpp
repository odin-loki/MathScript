#include <gtest/gtest.h>
#include "ms/runtime/dispatch.hpp"
#include "ms/runtime/topology.hpp"

using namespace ms;

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
