#include <gtest/gtest.h>
#include "ms/cuda/nccl.hpp"

using namespace ms::cuda;

TEST(NcclStubTest, unavailable_by_default) {
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (!nccl_available()) {
        EXPECT_EQ(nccl_device_count(), 0);
        EXPECT_EQ(nccl_comm_size(), 1u);
    }
#else
    EXPECT_FALSE(nccl_available());
    EXPECT_EQ(nccl_device_count(), 0);
    EXPECT_EQ(nccl_comm_size(), 1u);
#endif
}

TEST(NcclStubTest, allreduce_sum_identity) {
    EXPECT_DOUBLE_EQ(allreduce_sum(3.5), 3.5);
    EXPECT_DOUBLE_EQ(allreduce_sum(0.0), 0.0);
    EXPECT_DOUBLE_EQ(allreduce_sum(-2.25), -2.25);
}

TEST(NcclStubTest, allreduce_sum_idempotent) {
    const double once = allreduce_sum(42.0);
    const double twice = allreduce_sum(once);
    EXPECT_DOUBLE_EQ(once, twice);
}

TEST(NcclStubTest, comm_size_at_least_one) {
    EXPECT_GE(nccl_comm_size(), 1u);
}
