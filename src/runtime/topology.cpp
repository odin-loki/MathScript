// MathScript Runtime Topology Implementation

#include "ms/runtime/topology.hpp"
#include "ms/cuda/buffer.hpp"
#include <thread>

namespace ms {

SystemTopology detect_topology() {
    SystemTopology topo;
    const unsigned hw = std::thread::hardware_concurrency();
    const size_t threads = hw == 0 ? 1 : static_cast<size_t>(hw);
    topo.cpu_cores = {{}};
    topo.cpu_threads = {{}};
    for (size_t i = 0; i < threads; ++i) {
        topo.cpu_cores[0].push_back(i);
        topo.cpu_threads[0].push_back(i);
    }
    topo.numa_nodes = {0};
    topo.total_gpus = cuda::device_count();
    if (topo.total_gpus > 0) {
        for (int i = 0; i < topo.total_gpus; ++i) {
            topo.gpu_devices.push_back(cuda::device_name(i));
        }
    }
    return topo;
}

int nearest_numa_node(size_t, const SystemTopology&) {
    return 0;
}

bool has_cuda() {
    return cuda::available();
}

int get_gpu_count() {
    return cuda::device_count();
}

std::string get_gpu_model(int device) {
    return cuda::device_name(device);
}

} // namespace ms
