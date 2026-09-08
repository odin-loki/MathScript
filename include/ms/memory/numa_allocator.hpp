// MathScript NUMA Allocator - NUMA-local memory allocation

#pragma once

#include <cstddef>
#include <fstream>
#include <memory>
#include <new>
#include <string>
#include <vector>
#include "ms/memory/aligned_allocator.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// NUMA-local allocation needs a platform allocator. Define MS_HAVE_LIBNUMA (and
// link -lnuma) to enable it on Linux; Windows uses VirtualAllocExNuma directly.
// Without one of those, NumaTopology::available() is false and NumaAllocator
// degrades to AlignedAllocator, which is a correct allocator on any machine.
#if defined(MS_HAVE_LIBNUMA)
#include <numa.h>
#endif

#if defined(__linux__) && defined(__GLIBC__)
#include <sched.h>
#endif

namespace ms::memory {

// NUMA topology probe.
//
// num_nodes() and node_for_cpu() report what the platform exposes and work on any
// build. available() is narrower: it is true only when this build can actually
// perform a NUMA-local allocation (MS_HAVE_LIBNUMA on Linux, or Windows) AND the
// machine has more than one node. NumaAllocator keys its fast path off available()
// and falls back to AlignedAllocator otherwise.
class NumaTopology {
public:
    static NumaTopology& instance() {
        static NumaTopology topology;
        return topology;
    }

    NumaTopology(const NumaTopology&) = delete;
    NumaTopology& operator=(const NumaTopology&) = delete;

    // NUMA node of the CPU the calling thread is running on, or the first known
    // node when that cannot be determined.
    int nearest_node() const {
#if defined(__linux__) && defined(__GLIBC__)
        const int cpu = ::sched_getcpu();
        if (cpu >= 0) {
            const int node = node_for_cpu(cpu);
            if (node >= 0) {
                return node;
            }
        }
#endif
        return nodes_.empty() ? 0 : nodes_.front();
    }

    // NUMA node hosting the given logical CPU, or the first known node when the
    // CPU is not in the map.
    int node_for_cpu(int cpu_id) const {
        if (cpu_id >= 0 && static_cast<std::size_t>(cpu_id) < cpu_to_node_.size()) {
            const int node = cpu_to_node_[static_cast<std::size_t>(cpu_id)];
            if (node >= 0) {
                return node;
            }
        }
        return nodes_.empty() ? 0 : nodes_.front();
    }

    int num_nodes() const {
        return nodes_.empty() ? 1 : static_cast<int>(nodes_.size());
    }

    bool available() const {
#if defined(_WIN32) || defined(MS_HAVE_LIBNUMA)
        return nodes_.size() > 1;
#else
        return false;
#endif
    }

    // Free memory on the given node, in bytes. 0 when unknown.
    size_t node_memory_free(int node) const {
#if defined(_WIN32)
        ULONGLONG bytes = 0;
        if (node >= 0 && GetNumaAvailableMemoryNode(static_cast<UCHAR>(node), &bytes)) {
            return static_cast<size_t>(bytes);
        }
        return 0;
#elif defined(__linux__)
        if (node < 0) {
            return 0;
        }
        std::ifstream in("/sys/devices/system/node/node" + std::to_string(node) + "/meminfo");
        if (!in.is_open()) {
            return 0;
        }
        std::string line;
        while (std::getline(in, line)) {
            const std::size_t key = line.find("MemFree:");
            if (key == std::string::npos) {
                continue;
            }
            std::size_t pos = key + 8;
            while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) {
                ++pos;
            }
            size_t kb = 0;
            bool any = false;
            while (pos < line.size() && line[pos] >= '0' && line[pos] <= '9') {
                kb = kb * 10 + static_cast<size_t>(line[pos] - '0');
                ++pos;
                any = true;
            }
            return any ? kb * 1024 : 0;
        }
        return 0;
#else
        (void)node;
        return 0;
#endif
    }

private:
    NumaTopology() {
#if defined(__linux__)
        probe_sysfs();
#elif defined(_WIN32)
        ULONG highest = 0;
        if (GetNumaHighestNodeNumber(&highest)) {
            for (ULONG n = 0; n <= highest; ++n) {
                nodes_.push_back(static_cast<int>(n));
            }
        }
#endif
        if (nodes_.empty()) {
            nodes_.push_back(0);
        }
    }

#if defined(__linux__)
    static std::vector<int> parse_id_list(const std::string& text) {
        std::vector<int> ids;
        std::size_t pos = 0;
        while (pos <= text.size()) {
            const std::size_t comma = text.find(',', pos);
            const std::string token =
                text.substr(pos, (comma == std::string::npos) ? std::string::npos : comma - pos);
            if (!token.empty()) {
                const std::size_t dash = token.find('-');
                if (dash == std::string::npos) {
                    ids.push_back(to_int(token));
                } else {
                    const int lo = to_int(token.substr(0, dash));
                    const int hi = to_int(token.substr(dash + 1));
                    for (int v = lo; v >= 0 && v <= hi; ++v) {
                        ids.push_back(v);
                    }
                }
            }
            if (comma == std::string::npos) {
                break;
            }
            pos = comma + 1;
        }
        return ids;
    }

    static int to_int(const std::string& token) {
        int value = 0;
        bool any = false;
        for (const char ch : token) {
            if (ch < '0' || ch > '9') {
                if (any) {
                    break;
                }
                continue;
            }
            value = value * 10 + (ch - '0');
            any = true;
        }
        return any ? value : -1;
    }

    void probe_sysfs() {
        std::ifstream online("/sys/devices/system/node/online");
        std::string spec;
        if (!online.is_open() || !std::getline(online, spec)) {
            return;
        }
        const std::vector<int> node_ids = parse_id_list(spec);
        for (const int node : node_ids) {
            if (node < 0) {
                continue;
            }
            nodes_.push_back(node);
            std::ifstream cpulist("/sys/devices/system/node/node" + std::to_string(node) +
                                  "/cpulist");
            std::string cpus;
            if (!cpulist.is_open() || !std::getline(cpulist, cpus)) {
                continue;
            }
            for (const int cpu : parse_id_list(cpus)) {
                if (cpu < 0) {
                    continue;
                }
                if (static_cast<std::size_t>(cpu) >= cpu_to_node_.size()) {
                    cpu_to_node_.resize(static_cast<std::size_t>(cpu) + 1, -1);
                }
                cpu_to_node_[static_cast<std::size_t>(cpu)] = node;
            }
        }
    }
#endif

    std::vector<int> nodes_;
    std::vector<int> cpu_to_node_;
};

template<typename T>
class NumaAllocator {
    int node_;
public:
    using value_type = T;

    explicit NumaAllocator(int node = -1)
        : node_(node == -1
            ? NumaTopology::instance().nearest_node()
            : node) {}

    [[nodiscard]] T* allocate(size_t n) {
        if (!NumaTopology::instance().available()) {
            // Fall back to aligned allocator on non-NUMA systems
            return AlignedAllocator<T, 64>::allocate(n);
        }
#if defined(_WIN32)
        void* ptr = VirtualAllocExNuma(GetCurrentProcess(),
            nullptr, n * sizeof(T),
            MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, static_cast<DWORD>(node_));
#elif defined(MS_HAVE_LIBNUMA)
        void* ptr = numa_alloc_onnode(n * sizeof(T), node_);
#else
        // available() is false in this configuration, so this is unreachable; it
        // exists so the template still compiles when instantiated.
        void* ptr = nullptr;
#endif
        if (!ptr) {
            // NUMA-local allocation failed; fall back to aligned heap (no exceptions).
            return AlignedAllocator<T, 64>::allocate(n);
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_t n) noexcept {
        if (!ptr) return;
        if (!NumaTopology::instance().available()) {
            AlignedAllocator<T, 64>::deallocate(ptr, n);
            return;
        }
#if defined(_WIN32)
        VirtualFree(ptr, 0, MEM_RELEASE);
#elif defined(MS_HAVE_LIBNUMA)
        numa_free(ptr, n * sizeof(T));
#else
        AlignedAllocator<T, 64>::deallocate(ptr, n);
#endif
    }

    template<typename U>
    bool operator==(const NumaAllocator<U>& o) const noexcept {
        return node_ == o.node();
    }

    int node() const { return node_; }
};

} // namespace ms::memory
