// Aligned Memory Allocator
// Compatible with both Linux and Windows

#pragma once

#include <cstdlib>

namespace ms::memory {

// Free function wrappers for cross-platform compatibility
static inline void* aligned_alloc(size_t align, size_t size) {
    #ifdef _WIN32
    return _aligned_malloc(size, align);  // _aligned_malloc(size, alignment)
    #else
    return std::aligned_alloc(align, size);
    #endif
}

static inline void aligned_free(void* ptr) {
    #ifdef _WIN32
    _aligned_free(ptr);
    #else
    std::free(ptr);
    #endif
}

template<typename T, long long Align = alignof(T)>
class AlignedAllocator {
public:
    using value_type = T;
    
    static value_type* allocate(size_t n) {
        void* ptr = aligned_alloc(Align, n * sizeof(T));
        return static_cast<value_type*>(ptr);
    }
    
    static void deallocate(value_type* ptr, size_t n) {
        aligned_free(ptr);
    }
};

} // namespace ms::memory