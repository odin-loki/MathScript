// MathScript Runtime Topology Implementation

#include "ms/runtime/topology.hpp"
#include "ms/cuda/buffer.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace ms {

namespace {

std::optional<std::string> read_first_line(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return std::nullopt;
    }
    std::string line;
    if (!std::getline(in, line)) {
        return std::nullopt;
    }
    return line;
}

std::optional<long> parse_long(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) {
        text.remove_prefix(1);
    }
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\r')) {
        text.remove_suffix(1);
    }
    if (text.empty()) {
        return std::nullopt;
    }
    long value = 0;
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        return std::nullopt;
    }
    return value;
}

// Parse a sysfs CPU/node list such as "0-3,8,10-11" into individual ids.
std::vector<size_t> parse_id_list(std::string_view text) {
    std::vector<size_t> ids;
    while (!text.empty()) {
        const std::size_t comma = text.find(',');
        std::string_view token = (comma == std::string_view::npos) ? text : text.substr(0, comma);
        text = (comma == std::string_view::npos) ? std::string_view{} : text.substr(comma + 1);

        const std::size_t dash = token.find('-');
        if (dash == std::string_view::npos) {
            const auto single = parse_long(token);
            if (single && *single >= 0) {
                ids.push_back(static_cast<size_t>(*single));
            }
            continue;
        }
        const auto lo = parse_long(token.substr(0, dash));
        const auto hi = parse_long(token.substr(dash + 1));
        if (!lo || !hi || *lo < 0 || *hi < *lo) {
            continue;
        }
        for (long v = *lo; v <= *hi; ++v) {
            ids.push_back(static_cast<size_t>(v));
        }
    }
    return ids;
}

std::string cpu_topology_path(size_t cpu, const char* leaf) {
    return "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/" + leaf;
}

std::vector<size_t> online_cpus() {
    const auto online = read_first_line("/sys/devices/system/cpu/online");
    if (online) {
        return parse_id_list(*online);
    }
    return {};
}

// Fill numa_nodes and cpu_numa_node from /sys/devices/system/node. Returns false
// when the node hierarchy is not present.
bool probe_numa(SystemTopology& topo, size_t max_cpu_id) {
    const auto online = read_first_line("/sys/devices/system/node/online");
    if (!online) {
        return false;
    }
    const std::vector<size_t> nodes = parse_id_list(*online);
    if (nodes.empty()) {
        return false;
    }

    topo.cpu_numa_node.assign(max_cpu_id + 1, -1);
    for (const size_t node : nodes) {
        topo.numa_nodes.push_back(static_cast<int>(node));
        const auto cpulist =
            read_first_line("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
        if (!cpulist) {
            continue;
        }
        for (const size_t cpu : parse_id_list(*cpulist)) {
            if (cpu < topo.cpu_numa_node.size()) {
                topo.cpu_numa_node[cpu] = static_cast<int>(node);
            }
        }
    }
    return true;
}

// Group the online logical CPUs into sockets and physical cores using sysfs.
// Returns false when sysfs CPU topology is unavailable, leaving topo untouched.
bool probe_cpu_topology(SystemTopology& topo) {
    const std::vector<size_t> cpus = online_cpus();
    if (cpus.empty()) {
        return false;
    }

    struct Socket {
        long package = 0;
        std::vector<size_t> threads;
        std::vector<size_t> core_reps;
        std::vector<long> core_ids;
    };
    std::vector<Socket> sockets;
    size_t max_cpu_id = 0;
    bool any_package = false;

    for (const size_t cpu : cpus) {
        max_cpu_id = (std::max)(max_cpu_id, cpu);

        const auto package_text = read_first_line(cpu_topology_path(cpu, "physical_package_id"));
        const auto core_text = read_first_line(cpu_topology_path(cpu, "core_id"));
        if (!package_text) {
            continue;
        }
        const auto package = parse_long(*package_text);
        if (!package) {
            continue;
        }
        any_package = true;
        // core_id is only unique within a package; when it is missing, treat the
        // logical CPU as its own physical core.
        const long core = core_text ? parse_long(*core_text).value_or(static_cast<long>(cpu))
                                    : static_cast<long>(cpu);

        auto it = std::find_if(sockets.begin(), sockets.end(), [&](const Socket& s) {
            return s.package == *package;
        });
        if (it == sockets.end()) {
            sockets.push_back(Socket{*package, {}, {}, {}});
            it = sockets.end() - 1;
        }
        it->threads.push_back(cpu);
        if (std::find(it->core_ids.begin(), it->core_ids.end(), core) == it->core_ids.end()) {
            it->core_ids.push_back(core);
            it->core_reps.push_back(cpu);
        }
    }

    if (!any_package || sockets.empty()) {
        return false;
    }

    std::sort(sockets.begin(), sockets.end(), [](const Socket& a, const Socket& b) {
        return a.package < b.package;
    });

    for (const Socket& socket : sockets) {
        topo.cpu_sockets.push_back(std::to_string(socket.package));
        topo.cpu_cores.push_back(socket.core_reps);
        topo.cpu_threads.push_back(socket.threads);
    }

    if (!probe_numa(topo, max_cpu_id)) {
        topo.numa_nodes = {0};
        topo.cpu_numa_node.assign(max_cpu_id + 1, 0);
    }
    return true;
}

// Documented fallback: one synthetic socket, no SMT information, one NUMA node.
void synthesize_fallback(SystemTopology& topo) {
    const unsigned hw = std::thread::hardware_concurrency();
    const size_t threads = (hw == 0) ? size_t{1} : static_cast<size_t>(hw);

    topo.cpu_sockets = {std::string("0")};
    topo.cpu_cores.assign(1, {});
    topo.cpu_threads.assign(1, {});
    topo.cpu_numa_node.assign(threads, 0);
    for (size_t i = 0; i < threads; ++i) {
        topo.cpu_cores[0].push_back(i);
        topo.cpu_threads[0].push_back(i);
    }
    topo.numa_nodes = {0};
}

} // namespace

namespace {

// The actual probe. Opens /sys/devices/system/cpu/online, a couple of files per
// online CPU, and the NUMA cpulists: roughly 30 microseconds on a small machine
// and more as the CPU count grows.
SystemTopology probe_topology_now() {
    SystemTopology topo;
    topo.total_gpus = 0;

#if defined(__linux__)
    topo.detected = probe_cpu_topology(topo);
#endif
    if (!topo.detected) {
        synthesize_fallback(topo);
    }

    topo.total_gpus = cuda::device_count();
    if (topo.total_gpus > 0) {
        for (int i = 0; i < topo.total_gpus; ++i) {
            topo.gpu_devices.push_back(cuda::device_name(i));
        }
    }
    return topo;
}

} // namespace

const SystemTopology& cached_topology() {
    // The machine's CPU and NUMA layout does not change while the process runs,
    // so the sysfs walk is done once. This matters because it sits on a hot
    // path: decide() consults the topology on every dispatched operation, so an
    // uncached probe put its full cost on each one -- an fft() of 256 points
    // spent more time reading sysfs than transforming. Initialisation of a
    // function-local static is thread-safe.
    static const SystemTopology cached = probe_topology_now();
    return cached;
}

SystemTopology detect_topology() {
    return cached_topology();
}

int nearest_numa_node(size_t core_id, const SystemTopology& topo) {
    if (core_id < topo.cpu_numa_node.size()) {
        const int node = topo.cpu_numa_node[core_id];
        if (node >= 0) {
            return node;
        }
    }
    if (!topo.numa_nodes.empty()) {
        return topo.numa_nodes.front();
    }
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
