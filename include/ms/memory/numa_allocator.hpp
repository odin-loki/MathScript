// MathScript NUMA Allocator - NUMA-local memory allocation

#pragma once

#include <memory>
#include <new>
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

namespace ms::memory {

class NumaTopology {
public:
    static NumaTopology& instance();

    int  nearest_node() const;
    int  node_for_cpu(int cpu_id) const;
    int  num_nodes() const;
    bool available() const;
    size_t node_memory_free(int node) const;

private:
    NumaTopology();
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
            MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, node_);
#else
        void* ptr = numa_alloc_onnode(n * sizeof(T), node_);
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
#else
        numa_free(ptr, n * sizeof(T));
#endif
    }

    template<typename U>
    bool operator==(const NumaAllocator<U>& o) const noexcept {
        return node_ == o.node_;
    }

    int node() const { return node_; }
};

} // namespace ms::memory