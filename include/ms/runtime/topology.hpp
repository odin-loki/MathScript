// MathScript Runtime Topology Header

#pragma once

#include <numeric>
#include <string>
#include <vector>
#include <cstdint>

namespace ms {

// System topology information.
//
// cpu_sockets[s] is the platform's identifier for socket s (the sysfs
// physical_package_id on Linux). cpu_cores[s] lists one logical CPU per PHYSICAL
// core on socket s; cpu_threads[s] lists every logical CPU on socket s. On an SMT
// machine total_cores() < total_threads(). cpu_numa_node maps a logical CPU id to
// its NUMA node and is what nearest_numa_node() looks up.
struct SystemTopology {
    std::vector<std::string> cpu_sockets;
    std::vector<std::vector<size_t>> cpu_cores;
    std::vector<std::vector<size_t>> cpu_threads;
    std::vector<int> numa_nodes;
    // Indexed by logical CPU id; entries are -1 where the node is unknown. Empty
    // when NUMA topology could not be detected at all.
    std::vector<int> cpu_numa_node;
    // False when platform topology probing was unavailable and the fields above
    // hold the documented single-socket / single-node fallback synthesis.
    bool detected = false;
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

// Probe the machine topology.
//
// On Linux this reads /sys/devices/system/cpu/cpu*/topology/{physical_package_id,
// core_id} and /sys/devices/system/node/node*/cpulist. When sysfs is unavailable
// (or on other platforms, where no probe is implemented) the result is the
// documented fallback: one synthetic socket "0" holding hardware_concurrency()
// logical CPUs, the same count of physical cores, and a single NUMA node 0, with
// SystemTopology::detected set to false.
//
// The probe runs once per process and the result is reused; the machine's CPU
// and NUMA layout cannot change underneath a running process. Returns a copy --
// prefer cached_topology() on a hot path, which hands back a reference.
SystemTopology detect_topology();

// The same probed topology, by reference and without the copy. Use this where a
// topology is consulted per operation, such as dispatch decisions.
const SystemTopology& cached_topology();

// NUMA node hosting the given logical CPU. Returns the first known node (0 when
// the topology carries none) if core_id is not present in the CPU-to-node map.
int nearest_numa_node(size_t core_id, const SystemTopology& topo);

// Check GPU availability
bool has_cuda();
int get_gpu_count();
std::string get_gpu_model(int device);

} // namespace ms