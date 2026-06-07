// MathScript Runtime Topology Header

#pragma once

#include <numeric>
#include <string>
#include <vector>
#include <cstdint>

namespace ms {

// System topology information
struct SystemTopology {
    std::vector<std::string> cpu_sockets;
    std::vector<std::vector<size_t>> cpu_cores;
    std::vector<std::vector<size_t>> cpu_threads;
    std::vector<int> numa_nodes;
    int total_gpus;
    std::vector<std::string> gpu_devices;
    
    size_t total_cores() const {
        size_t total = 0;
        for (const auto& core : cpu_cores) {
            total += core.size();
        }
        return total;
    }

    size_t total_threads() const {
        size_t total = 0;
        for (const auto& thread : cpu_threads) {
            total += thread.size();
        }
        return total;
    }
    
    size_t numa_count() const {
        return numa_nodes.empty() ? 1 : numa_nodes.size();
    }
};

// Initialize topology detection
SystemTopology detect_topology();

// Get nearest NUMA node
int nearest_numa_node(size_t core_id, const SystemTopology& topo);

// Check GPU availability
bool has_cuda();
int get_gpu_count();
std::string get_gpu_model(int device);

} // namespace ms