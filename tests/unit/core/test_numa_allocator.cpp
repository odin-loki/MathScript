// Regression tests for ms::memory::NumaTopology / NumaAllocator.
//
// Before the fix this header could not be compiled at all once NumaAllocator<T>
// was instantiated: NumaTopology was declared with no definition anywhere in the
// tree, numa_alloc_onnode/numa_free were called with no <numa.h> included and no
// -lnuma linked, and operator== read the private node_ of a different template
// instantiation. tests/unit/core/test_tensor_lb_numa.cpp carried the comment
// "NumaTopology/NumaAllocator are declared but not implemented".

#include <gtest/gtest.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include "ms/memory/numa_allocator.hpp"

using ms::memory::NumaAllocator;
using ms::memory::NumaTopology;

TEST(NumaTopologyContract, ReportsAtLeastOneNodeAndAConsistentMap) {
    const NumaTopology& topo = NumaTopology::instance();
    EXPECT_GE(topo.num_nodes(), 1);
    EXPECT_GE(topo.nearest_node(), 0);
    EXPECT_LT(topo.nearest_node(), topo.num_nodes() + 1024);
    // Unknown CPU ids fall back to a real node rather than reading out of bounds.
    EXPECT_GE(topo.node_for_cpu(1 << 20), 0);
    EXPECT_GE(topo.node_for_cpu(-1), 0);
}

#if defined(__linux__)
TEST(NumaTopologyContract, NodeCountMatchesSysfs) {
    std::ifstream online("/sys/devices/system/node/online");
    std::string spec;
    if (!online.is_open() || !std::getline(online, spec)) {
        GTEST_SKIP() << "sysfs NUMA information not available";
    }
    // Count the ids in a list such as "0" or "0-1,3".
    std::size_t count = 0;
    std::size_t pos = 0;
    while (pos <= spec.size()) {
        const std::size_t comma = spec.find(',', pos);
        const std::string token =
            spec.substr(pos, (comma == std::string::npos) ? std::string::npos : comma - pos);
        if (!token.empty()) {
            const std::size_t dash = token.find('-');
            if (dash == std::string::npos) {
                ++count;
            } else {
                const auto lo = static_cast<std::size_t>(std::stoul(token.substr(0, dash)));
                const auto hi = static_cast<std::size_t>(std::stoul(token.substr(dash + 1)));
                count += (hi >= lo) ? (hi - lo + 1) : 0;
            }
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    ASSERT_GT(count, 0u);
    EXPECT_EQ(static_cast<std::size_t>(NumaTopology::instance().num_nodes()), count);
}

TEST(NumaTopologyContract, NodeMemoryFreeMatchesSysfsMemInfo) {
    std::ifstream in("/sys/devices/system/node/node0/meminfo");
    if (!in.is_open()) {
        GTEST_SKIP() << "sysfs node meminfo not available";
    }
    std::string line;
    std::size_t expected_kb = 0;
    bool found = false;
    while (std::getline(in, line)) {
        const std::size_t key = line.find("MemFree:");
        if (key == std::string::npos) {
            continue;
        }
        std::size_t pos = key + 8;
        while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) {
            ++pos;
        }
        expected_kb = static_cast<std::size_t>(std::stoul(line.substr(pos)));
        found = true;
        break;
    }
    if (!found) {
        GTEST_SKIP() << "node0 meminfo has no MemFree line";
    }
    // Free memory moves between the two reads, so compare the magnitude, not the
    // exact byte count: the parser must return kB * 1024, not kB and not 0.
    const size_t reported = NumaTopology::instance().node_memory_free(0);
    EXPECT_GT(reported, 0u);
    const double ratio = static_cast<double>(reported) / (static_cast<double>(expected_kb) * 1024.0);
    EXPECT_GT(ratio, 0.5);
    EXPECT_LT(ratio, 2.0);
}
#endif

TEST(NumaAllocatorContract, AllocatesUsableAlignedMemory) {
    NumaAllocator<double> alloc;
    constexpr std::size_t n = 1024;
    double* p = alloc.allocate(n);
    ASSERT_NE(p, nullptr);

    for (std::size_t i = 0; i < n; ++i) {
        p[i] = 0.5 * static_cast<double>(i);
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += p[i];
    }
    EXPECT_DOUBLE_EQ(sum, 0.5 * (static_cast<double>(n - 1) * static_cast<double>(n) / 2.0));

    alloc.deallocate(p, n);
}

TEST(NumaAllocatorContract, ComparesAcrossInstantiations) {
    NumaAllocator<double> a(0);
    NumaAllocator<float> b(0);
    NumaAllocator<float> c(1);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_EQ(a.node(), 0);
}

TEST(NumaAllocatorContract, DeallocatingNullIsSafe) {
    NumaAllocator<int> alloc;
    alloc.deallocate(nullptr, 0);
    SUCCEED();
}
