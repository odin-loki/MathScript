// MathScript Memory Policy Header

#pragma once

#include <cstdint>

namespace ms::memory {

// Memory policy flags
enum class PolicyFlag : uint32_t {
    NONE = 0,
    ALIGN_64 = 1 << 0,
    ALIGN_128 = 1 << 1,
    PINNED = 1 << 2,
    NUMA = 1 << 3,
    ARBITRARY = 1 << 4,
    POOLED = 1 << 5
};

// Get aligned allocation size
inline size_t aligned_size(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

} // namespace ms::memory