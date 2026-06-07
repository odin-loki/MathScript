// MathScript Arena Allocator - Per-thread monotonic arena

#pragma once

#include <memory_resource>
#include <thread>

namespace ms::memory {

class Arena {
    static constexpr size_t DEFAULT_CAPACITY = 64 * 1024 * 1024; // 64 MB

    std::pmr::monotonic_buffer_resource resource_;
    std::pmr::polymorphic_allocator<std::byte> alloc_;
    size_t capacity_;

public:
    explicit Arena(size_t capacity = DEFAULT_CAPACITY)
        : resource_(capacity, std::pmr::get_default_resource())
        , alloc_(&resource_)
        , capacity_(capacity) {}

    template<typename T>
    T* allocate(size_t n = 1) {
        return alloc_.allocate_object<T>(n);
    }

    template<typename T, typename... Args>
    T* construct(Args&&... args) {
        T* ptr = allocate<T>();
        std::construct_at(ptr, std::forward<Args>(args)...);
        return ptr;
    }

    void reset() noexcept {
        resource_.release();
    }

    size_t capacity() const { return capacity_; }
};

// Per-thread arena — zero contention, no locking
inline thread_local Arena scratch_arena;

// RAII scope guard — resets arena on destruction
struct [[nodiscard]] ScratchScope {
    ~ScratchScope() { scratch_arena.reset(); }
};

} // namespace ms::memory