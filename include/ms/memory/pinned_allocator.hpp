// Pinned Memory Allocator
// For CPU-only builds

#pragma once

#include <cstdlib>

namespace ms::memory {

class PinnedAllocator {
public:
    using value_type = char*;
    
    static value_type* allocate(size_t n) {
        return static_cast<value_type*>(std::malloc(n * sizeof(value_type)));
    }
    
    static void deallocate(value_type* ptr, size_t n) {
        std::free(ptr);
    }
};

} // namespace ms::memory