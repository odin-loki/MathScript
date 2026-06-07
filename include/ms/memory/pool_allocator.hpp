// MathScript Pool Allocator - Slab-based for small objects

#pragma once

#include <mutex>
#include <cstdlib>

namespace ms::memory {

template<size_t BlockSize, size_t BlocksPerSlab = 4096>
class PoolAllocator {
    struct Slab {
        alignas(64) std::byte data[BlockSize * BlocksPerSlab];
        Slab* next;
    };

    Slab*  slabs_    = nullptr;
    void*  freelist_ = nullptr;
    size_t allocated_ = 0;
    std::mutex mtx_;

public:
    void* allocate() {
        std::lock_guard lock(mtx_);
        if (!freelist_) grow();
        void* ptr = freelist_;
        freelist_ = *static_cast<void**>(freelist_);
        ++allocated_;
        return ptr;
    }

    void deallocate(void* ptr) noexcept {
        std::lock_guard lock(mtx_);
        *static_cast<void**>(ptr) = freelist_;
        freelist_ = ptr;
        --allocated_;
    }

    size_t allocated() const { return allocated_; }

private:
    void grow() {
        auto* slab = new Slab();
        slab->next = slabs_;
        slabs_ = slab;
        // Chain all blocks into the free list
        for (size_t i = 0; i < BlocksPerSlab; ++i) {
            void* block = slab->data + i * BlockSize;
            *static_cast<void**>(block) = freelist_;
            freelist_ = block;
        }
    }
};

// Pre-defined sizes for common symbolic node sizes
extern PoolAllocator<16>  pool16;
extern PoolAllocator<32>  pool32;
extern PoolAllocator<64>  pool64;
extern PoolAllocator<128> pool128;

} // namespace ms::memory