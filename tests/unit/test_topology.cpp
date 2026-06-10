#include <gtest/gtest.h>
#include "ms/runtime/topology.hpp"

using namespace ms;

TEST(TopologyTest, detect_returns_valid) {
    const SystemTopology topo = detect_topology();
    const size_t total_cores = topo.total_cores();
    EXPECT_GE(total_cores, 1u);
}

TEST(TopologyTest, total_threads_positive) {
    const SystemTopology topo = detect_topology();
    EXPECT_GE(topo.total_threads(), 1u);
}

TEST(TopologyTest, has_cuda_no_throw) {
    EXPECT_NO_THROW({ (void)has_cuda(); });
}

TEST(TopologyTest, nearest_numa_node) {
    const SystemTopology topo = detect_topology();
    const int node = nearest_numa_node(0, topo);
    EXPECT_GE(node, 0);
    EXPECT_LT(static_cast<size_t>(node), topo.numa_count());
}
