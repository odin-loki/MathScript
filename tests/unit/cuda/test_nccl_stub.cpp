#include <gtest/gtest.h>
#include "ms/cuda/buffer.hpp"
#include "ms/cuda/nccl.hpp"
#include "ms/interp/repl_engine.hpp"

#include <cstddef>
#include <span>
#include <variant>
#include <vector>

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

// ---------------------------------------------------------------------------
// NCCL collectives: topology, lifetime, host vectors and device buffers.
// Every assertion below holds in the stub build (communicator size 1) and on
// real hardware, because the buffer collectives span exactly the buffers the
// caller hands over regardless of how many GPUs are present.
// ---------------------------------------------------------------------------

namespace {

DeviceBuffer make_filled(const std::vector<double>& values) {
    DeviceBuffer buf = make_device_buffer(values.size() * sizeof(double));
    copy_host_to_device(values.data(), buf, values.size() * sizeof(double));
    return buf;
}

std::vector<double> read_back(const DeviceBuffer& buf, size_t count) {
    std::vector<double> out(count, 0.0);
    copy_device_to_host(buf, out.data(), count * sizeof(double));
    return out;
}

} // namespace

TEST(NcclCollectivesTest, topology_reports_single_rank) {
#if !defined(MS_HAS_NCCL) || !MS_HAS_NCCL
    EXPECT_FALSE(nccl_available());
    EXPECT_EQ(nccl_device_count(), 0);
    EXPECT_EQ(nccl_comm_size(), 1u);
    EXPECT_EQ(nccl_local_rank_count(), 0u);
    EXPECT_EQ(nccl_backend_name(), "stub");
#endif
    EXPECT_GE(nccl_comm_size(), 1u);
    EXPECT_GE(nccl_rank(), 0);
    EXPECT_FALSE(nccl_backend_name().empty());
}

TEST(NcclCollectivesTest, scalar_identity_preserved) {
    EXPECT_DOUBLE_EQ(allreduce_sum(3.5), 3.5);
    EXPECT_DOUBLE_EQ(allreduce_max(-2.25), -2.25);
    EXPECT_DOUBLE_EQ(allreduce_min(0.0), 0.0);
    EXPECT_DOUBLE_EQ(allreduce_prod(7.0), 7.0);
    EXPECT_DOUBLE_EQ(allreduce_avg(3.5), 3.5);
    EXPECT_DOUBLE_EQ(broadcast(3.5, 0), 3.5);
    EXPECT_DOUBLE_EQ(reduce(3.5, 0), 3.5);
    EXPECT_DOUBLE_EQ(allgather(3.5), 3.5);
    // An out-of-range root is a documented no-op, never a crash.
    EXPECT_DOUBLE_EQ(broadcast(3.5, 9), 3.5);
    EXPECT_DOUBLE_EQ(reduce(3.5, -1), 3.5);
}

TEST(NcclCollectivesTest, scalar_entry_points_are_idempotent) {
    EXPECT_DOUBLE_EQ(allreduce_sum(allreduce_sum(42.0)), allreduce_sum(42.0));
    EXPECT_DOUBLE_EQ(broadcast(broadcast(42.0)), broadcast(42.0));
    EXPECT_DOUBLE_EQ(allgather(allgather(42.0)), allgather(42.0));
}

TEST(NcclCollectivesTest, rank_bootstrap_validates_before_touching_nccl) {
    const std::vector<unsigned char> short_id(7, 0);
    const auto bad_size = nccl_init_rank(short_id, 0, 1, 0);
    ASSERT_FALSE(bad_size.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::DimensionMismatch>(bad_size.error()));

    const std::vector<unsigned char> id(128, 0);
    const auto bad_world = nccl_init_rank(id, 0, 0, 0);
    ASSERT_FALSE(bad_world.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::ValueOutOfRange>(bad_world.error()));

    const auto bad_rank = nccl_init_rank(id, 4, 2, 0);
    ASSERT_FALSE(bad_rank.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::ValueOutOfRange>(bad_rank.error()));

#if !defined(MS_HAS_NCCL) || !MS_HAS_NCCL
    const auto unique_id = nccl_make_unique_id();
    ASSERT_FALSE(unique_id.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::DeviceError>(unique_id.error()));

    const auto joined = nccl_init_rank(id, 0, 1, 0);
    ASSERT_FALSE(joined.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::DeviceError>(joined.error()));
#endif

    nccl_finalize();
    nccl_finalize();
    EXPECT_GE(nccl_comm_size(), 1u);
}

TEST(NcclCollectivesTest, host_collectives_identity_at_comm_size_one) {
    if (nccl_comm_size() != 1u) {
        GTEST_SKIP() << "multi-rank communicator active";
    }
    std::vector<double> values{1.0, 2.0, 3.0};
    const std::vector<double> expected = values;

    ASSERT_TRUE(allreduce_host(values, ReduceOp::Sum).has_value());
    EXPECT_EQ(values, expected);
    ASSERT_TRUE(broadcast_host(values, 0).has_value());
    EXPECT_EQ(values, expected);
    ASSERT_TRUE(reduce_host(values, ReduceOp::Sum, 0).has_value());
    EXPECT_EQ(values, expected);

    const auto gathered = allgather_host(values);
    ASSERT_TRUE(gathered.has_value());
    EXPECT_EQ(*gathered, expected);

    std::vector<double> empty;
    ASSERT_TRUE(allreduce_host(empty, ReduceOp::Max).has_value());
    EXPECT_TRUE(empty.empty());
}

TEST(NcclCollectivesTest, host_collectives_reject_out_of_range_root) {
    std::vector<double> values{1.0, 2.0};
    const auto too_high = broadcast_host(values, static_cast<int>(nccl_comm_size()));
    ASSERT_FALSE(too_high.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::ValueOutOfRange>(too_high.error()));

    const auto negative = reduce_host(values, ReduceOp::Sum, -1);
    ASSERT_FALSE(negative.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::ValueOutOfRange>(negative.error()));
}

TEST(NcclCollectivesTest, buffer_allreduce_sum_across_three_ranks) {
    DeviceBuffer a = make_filled({1.0, 2.0, 3.0});
    DeviceBuffer b = make_filled({10.0, 20.0, 30.0});
    DeviceBuffer c = make_filled({100.0, 200.0, 300.0});
    DeviceBuffer* ranks[] = {&a, &b, &c};

    ASSERT_TRUE(allreduce_buffer(ranks, 3, ReduceOp::Sum).has_value());

    const std::vector<double> expected{111.0, 222.0, 333.0};
    EXPECT_EQ(read_back(a, 3), expected);
    EXPECT_EQ(read_back(b, 3), expected);
    EXPECT_EQ(read_back(c, 3), expected);
}

TEST(NcclCollectivesTest, buffer_allreduce_all_ops) {
    {
        DeviceBuffer a = make_filled({2.0, 3.0});
        DeviceBuffer b = make_filled({5.0, 7.0});
        DeviceBuffer* ranks[] = {&a, &b};
        ASSERT_TRUE(allreduce_buffer(ranks, 2, ReduceOp::Prod).has_value());
        EXPECT_EQ(read_back(a, 2), (std::vector<double>{10.0, 21.0}));
    }
    {
        DeviceBuffer a = make_filled({2.0, 9.0});
        DeviceBuffer b = make_filled({5.0, 7.0});
        DeviceBuffer* ranks[] = {&a, &b};
        ASSERT_TRUE(allreduce_buffer(ranks, 2, ReduceOp::Max).has_value());
        EXPECT_EQ(read_back(b, 2), (std::vector<double>{5.0, 9.0}));
    }
    {
        DeviceBuffer a = make_filled({2.0, 9.0});
        DeviceBuffer b = make_filled({5.0, 7.0});
        DeviceBuffer* ranks[] = {&a, &b};
        ASSERT_TRUE(allreduce_buffer(ranks, 2, ReduceOp::Min).has_value());
        EXPECT_EQ(read_back(a, 2), (std::vector<double>{2.0, 7.0}));
    }
    {
        DeviceBuffer a = make_filled({2.0, 10.0});
        DeviceBuffer b = make_filled({4.0, 20.0});
        DeviceBuffer* ranks[] = {&a, &b};
        ASSERT_TRUE(allreduce_buffer(ranks, 2, ReduceOp::Avg).has_value());
        const std::vector<double> got = read_back(a, 2);
        ASSERT_EQ(got.size(), 2u);
        EXPECT_NEAR(got[0], 3.0, 1e-12);
        EXPECT_NEAR(got[1], 15.0, 1e-12);
    }
}

TEST(NcclCollectivesTest, buffer_broadcast_and_reduce_root_semantics) {
    DeviceBuffer a = make_filled({1.0, 1.0});
    DeviceBuffer b = make_filled({2.0, 2.0});
    DeviceBuffer c = make_filled({3.0, 3.0});
    DeviceBuffer* ranks[] = {&a, &b, &c};

    ASSERT_TRUE(broadcast_buffer(ranks, 2, 1).has_value());
    const std::vector<double> broadcasted{2.0, 2.0};
    EXPECT_EQ(read_back(a, 2), broadcasted);
    EXPECT_EQ(read_back(b, 2), broadcasted);
    EXPECT_EQ(read_back(c, 2), broadcasted);

    DeviceBuffer d = make_filled({1.0, 2.0});
    DeviceBuffer e = make_filled({10.0, 20.0});
    DeviceBuffer* pair[] = {&d, &e};
    ASSERT_TRUE(reduce_buffer(pair, 2, ReduceOp::Sum, 1).has_value());
    EXPECT_EQ(read_back(d, 2), (std::vector<double>{1.0, 2.0}));
    EXPECT_EQ(read_back(e, 2), (std::vector<double>{11.0, 22.0}));
}

TEST(NcclCollectivesTest, buffer_allgather_concatenates_rank_order) {
    DeviceBuffer s0 = make_filled({1.0, 2.0});
    DeviceBuffer s1 = make_filled({3.0, 4.0});
    DeviceBuffer r0 = make_filled({0.0, 0.0, 0.0, 0.0});
    DeviceBuffer r1 = make_filled({0.0, 0.0, 0.0, 0.0});
    DeviceBuffer* send[] = {&s0, &s1};
    DeviceBuffer* recv[] = {&r0, &r1};

    ASSERT_TRUE(allgather_buffer(send, recv, 2).has_value());

    const std::vector<double> expected{1.0, 2.0, 3.0, 4.0};
    EXPECT_EQ(read_back(r0, 4), expected);
    EXPECT_EQ(read_back(r1, 4), expected);
}

TEST(NcclCollectivesTest, buffer_degenerate_inputs) {
    DeviceBuffer a = make_filled({1.0, 2.0});
    DeviceBuffer b = make_filled({3.0});

    const auto empty = allreduce_buffer(std::span<DeviceBuffer* const>{}, 2, ReduceOp::Sum);
    ASSERT_FALSE(empty.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::DomainError>(empty.error()));

    DeviceBuffer* with_null[] = {&a, nullptr};
    const auto null_rank = allreduce_buffer(with_null, 2, ReduceOp::Sum);
    ASSERT_FALSE(null_rank.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::DomainError>(null_rank.error()));

    DeviceBuffer* short_rank[] = {&a, &b};
    const auto too_small = allreduce_buffer(short_rank, 2, ReduceOp::Sum);
    ASSERT_FALSE(too_small.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::DimensionMismatch>(too_small.error()));

    DeviceBuffer* single[] = {&a};
    ASSERT_TRUE(allreduce_buffer(single, 0, ReduceOp::Sum).has_value());
    EXPECT_EQ(read_back(a, 2), (std::vector<double>{1.0, 2.0}));
    if (nccl_comm_size() == 1u) {
        ASSERT_TRUE(allreduce_buffer(single, 2, ReduceOp::Avg).has_value());
        EXPECT_EQ(read_back(a, 2), (std::vector<double>{1.0, 2.0}));
    }

    DeviceBuffer c = make_filled({5.0, 6.0});
    DeviceBuffer* pair[] = {&a, &c};
    const auto bad_root = broadcast_buffer(pair, 2, 5);
    ASSERT_FALSE(bad_root.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::ValueOutOfRange>(bad_root.error()));

    DeviceBuffer* one_recv[] = {&c};
    const auto mismatched = allgather_buffer(pair, one_recv, 1);
    ASSERT_FALSE(mismatched.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::DimensionMismatch>(mismatched.error()));
}
