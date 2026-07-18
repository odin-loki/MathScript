#include <gtest/gtest.h>
#include "ms/cuda/nccl.hpp"
#include "ms/interp/repl_engine.hpp"

using namespace ms::cuda;
using namespace ms::interp;

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

TEST(NcclStubTest, allreduce_max_identity) {
    EXPECT_DOUBLE_EQ(allreduce_max(3.5), 3.5);
    EXPECT_DOUBLE_EQ(allreduce_max(0.0), 0.0);
    EXPECT_DOUBLE_EQ(allreduce_max(-2.25), -2.25);
}

TEST(NcclStubTest, allreduce_min_identity) {
    EXPECT_DOUBLE_EQ(allreduce_min(3.5), 3.5);
    EXPECT_DOUBLE_EQ(allreduce_min(0.0), 0.0);
    EXPECT_DOUBLE_EQ(allreduce_min(-2.25), -2.25);
}

TEST(NcclStubTest, allreduce_max_idempotent) {
    const double once = allreduce_max(42.0);
    const double twice = allreduce_max(once);
    EXPECT_DOUBLE_EQ(once, twice);
}

TEST(NcclStubTest, allreduce_min_idempotent) {
    const double once = allreduce_min(42.0);
    const double twice = allreduce_min(once);
    EXPECT_DOUBLE_EQ(once, twice);
}

TEST(NcclStubTest, allreduce_prod_identity) {
    EXPECT_DOUBLE_EQ(allreduce_prod(3.5), 3.5);
    EXPECT_DOUBLE_EQ(allreduce_prod(0.0), 0.0);
    EXPECT_DOUBLE_EQ(allreduce_prod(-2.25), -2.25);
}

TEST(NcclStubTest, allreduce_prod_idempotent) {
    const double once = allreduce_prod(42.0);
    const double twice = allreduce_prod(once);
    EXPECT_DOUBLE_EQ(once, twice);
}

TEST(NcclStubTest, allreduce_avg_identity) {
    EXPECT_DOUBLE_EQ(allreduce_avg(3.5), 3.5);
    EXPECT_DOUBLE_EQ(allreduce_avg(0.0), 0.0);
    EXPECT_DOUBLE_EQ(allreduce_avg(-2.25), -2.25);
}

TEST(NcclStubTest, allreduce_avg_idempotent) {
    const double once = allreduce_avg(42.0);
    const double twice = allreduce_avg(once);
    EXPECT_DOUBLE_EQ(once, twice);
}

TEST(NcclStubTest, broadcast_identity) {
    EXPECT_DOUBLE_EQ(broadcast(3.5), 3.5);
    EXPECT_DOUBLE_EQ(broadcast(0.0), 0.0);
    EXPECT_DOUBLE_EQ(broadcast(-2.25), -2.25);
    EXPECT_DOUBLE_EQ(broadcast(7.0, 0), 7.0);
}

TEST(NcclStubTest, broadcast_idempotent) {
    const double once = broadcast(42.0);
    const double twice = broadcast(once);
    EXPECT_DOUBLE_EQ(once, twice);
}

TEST(NcclStubTest, reduce_identity) {
    EXPECT_DOUBLE_EQ(reduce(3.5), 3.5);
    EXPECT_DOUBLE_EQ(reduce(0.0), 0.0);
    EXPECT_DOUBLE_EQ(reduce(-2.25), -2.25);
    EXPECT_DOUBLE_EQ(reduce(7.0, 0), 7.0);
}

TEST(NcclStubTest, reduce_idempotent) {
    const double once = reduce(42.0);
    const double twice = reduce(once);
    EXPECT_DOUBLE_EQ(once, twice);
}

TEST(NcclStubTest, allgather_identity) {
    EXPECT_DOUBLE_EQ(allgather(3.5), 3.5);
    EXPECT_DOUBLE_EQ(allgather(0.0), 0.0);
    EXPECT_DOUBLE_EQ(allgather(-2.25), -2.25);
}

TEST(NcclStubTest, allgather_idempotent) {
    const double once = allgather(42.0);
    const double twice = allgather(once);
    EXPECT_DOUBLE_EQ(once, twice);
}

TEST(NcclStubTest, repl_cuda_allgather) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("y = cuda_allgather(3.5)").has_value());
    ASSERT_GT(interp.state().scalars.count("y"), 0u);
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("y"), 3.5);
    const auto help = interp.execute("help");
    ASSERT_TRUE(help.has_value());
    EXPECT_NE(help->find("cuda_allgather(x)"), std::string::npos) << *help;
}

TEST(NcclStubTest, comm_size_at_least_one) {
    EXPECT_GE(nccl_comm_size(), 1u);
}

TEST(NcclStubTest, repl_nccl_introspect_queries) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("a = cuda_nccl_available()").has_value());
    ASSERT_TRUE(interp.execute("s = cuda_nccl_comm_size()").has_value());
    ASSERT_TRUE(interp.execute("d = cuda_nccl_device_count()").has_value());
#if !defined(MS_HAS_NCCL) || !MS_HAS_NCCL
    ASSERT_GT(interp.state().scalars.count("a"), 0u);
    ASSERT_GT(interp.state().scalars.count("s"), 0u);
    ASSERT_GT(interp.state().scalars.count("d"), 0u);
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("a"), 0.0);
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("s"), 1.0);
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("d"), 0.0);
#endif
    const auto help = interp.execute("help");
    ASSERT_TRUE(help.has_value());
    EXPECT_NE(help->find("cuda_nccl_available()"), std::string::npos) << *help;
    EXPECT_NE(help->find("cuda_nccl_comm_size()"), std::string::npos) << *help;
    EXPECT_NE(help->find("cuda_nccl_device_count()"), std::string::npos) << *help;
}
