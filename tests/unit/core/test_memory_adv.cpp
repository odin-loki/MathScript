// MathScript: Advanced Memory Allocator Tests
// Tests for Arena, PoolAllocator, AlignedAllocator, ScratchScope

#include <gtest/gtest.h>
#include <cstddef>
#include <vector>
#include <cstdint>
#include "ms/memory/arena.hpp"
#include "ms/memory/pool_allocator.hpp"
#include "ms/memory/aligned_allocator.hpp"

using namespace ms::memory;

// ---------------------------------------------------------------------------
// Arena allocator tests
// ---------------------------------------------------------------------------

TEST(MemoryAdv, Arena_DefaultCapacity) {
    Arena arena;
    EXPECT_GT(arena.capacity(), 0u);
}

TEST(MemoryAdv, Arena_CustomCapacity) {
    Arena arena(1024 * 1024);  // 1 MB
    EXPECT_EQ(arena.capacity(), 1024 * 1024u);
}

TEST(MemoryAdv, Arena_Allocate_Returns_NonNull) {
    Arena arena(64 * 1024);
    int* ptr = arena.allocate<int>(10);
    EXPECT_NE(ptr, nullptr);
}

TEST(MemoryAdv, Arena_Allocate_Multiple) {
    Arena arena(64 * 1024);
    int* p1 = arena.allocate<int>(4);
    double* p2 = arena.allocate<double>(4);
    EXPECT_NE(p1, nullptr);
    EXPECT_NE(p2, nullptr);
    EXPECT_NE(static_cast<void*>(p1), static_cast<void*>(p2));
}

TEST(MemoryAdv, Arena_Construct_Object) {
    Arena arena(64 * 1024);
    int* val = arena.construct<int>(42);
    ASSERT_NE(val, nullptr);
    EXPECT_EQ(*val, 42);
}

TEST(MemoryAdv, Arena_Construct_Double) {
    Arena arena(64 * 1024);
    double* val = arena.construct<double>(3.14);
    ASSERT_NE(val, nullptr);
    EXPECT_DOUBLE_EQ(*val, 3.14);
}

TEST(MemoryAdv, Arena_Reset_AllowsReuse) {
    Arena arena(64 * 1024);
    int* p1 = arena.allocate<int>(10);
    EXPECT_NE(p1, nullptr);
    arena.reset();
    // After reset, allocations should work again
    int* p2 = arena.allocate<int>(10);
    EXPECT_NE(p2, nullptr);
}

TEST(MemoryAdv, ScratchArena_IsThreadLocal) {
    // Thread-local scratch_arena should be accessible
    scratch_arena.reset();
    int* p = scratch_arena.allocate<int>(4);
    EXPECT_NE(p, nullptr);
    scratch_arena.reset();
}

TEST(MemoryAdv, ScratchScope_ResetsOnDestruction) {
    {
        ScratchScope scope;
        scratch_arena.allocate<int>(100);
        // When scope exits, arena is reset
    }
    // After scope exit, should still be able to allocate
    int* p = scratch_arena.allocate<int>(1);
    EXPECT_NE(p, nullptr);
    scratch_arena.reset();
}

// ---------------------------------------------------------------------------
// PoolAllocator tests
// ---------------------------------------------------------------------------

TEST(MemoryAdv, Pool_Allocate_Returns_NonNull) {
    PoolAllocator<32> pool;
    void* p = pool.allocate();
    EXPECT_NE(p, nullptr);
    pool.deallocate(p);
}

TEST(MemoryAdv, Pool_AllocateAndFree_Smoke) {
    PoolAllocator<64, 256> pool;
    std::vector<void*> ptrs;
    for (int i = 0; i < 100; ++i) {
        ptrs.push_back(pool.allocate());
        EXPECT_NE(ptrs.back(), nullptr);
    }
    for (void* p : ptrs) pool.deallocate(p);
}

TEST(MemoryAdv, Pool_Reuse_FreedBlock) {
    PoolAllocator<16, 64> pool;
    void* p1 = pool.allocate();
    pool.deallocate(p1);
    void* p2 = pool.allocate();
    // p2 might be p1 (reused) or a new block; either is valid
    EXPECT_NE(p2, nullptr);
    pool.deallocate(p2);
}

TEST(MemoryAdv, Pool_AllocationCount) {
    PoolAllocator<32> pool;
    EXPECT_EQ(pool.allocated(), 0u);
    void* p1 = pool.allocate();
    EXPECT_EQ(pool.allocated(), 1u);
    void* p2 = pool.allocate();
    EXPECT_EQ(pool.allocated(), 2u);
    pool.deallocate(p1);
    pool.deallocate(p2);
}

TEST(MemoryAdv, Pool_AllocatedDecreases_OnFree) {
    PoolAllocator<128> pool;
    void* p = pool.allocate();
    EXPECT_EQ(pool.allocated(), 1u);
    pool.deallocate(p);
    EXPECT_EQ(pool.allocated(), 0u);
}

// ---------------------------------------------------------------------------
// AlignedAllocator tests
// ---------------------------------------------------------------------------

TEST(MemoryAdv, AlignedAlloc_Allocate_NonNull) {
    double* p = AlignedAllocator<double, 64>::allocate(16);
    EXPECT_NE(p, nullptr);
    AlignedAllocator<double, 64>::deallocate(p, 16);
}

TEST(MemoryAdv, AlignedAlloc_CorrectAlignment) {
    double* p = AlignedAllocator<double, 32>::allocate(8);
    ASSERT_NE(p, nullptr);
    uintptr_t addr = reinterpret_cast<uintptr_t>(p);
    EXPECT_EQ(addr % 32, 0u) << "Address not aligned to 32 bytes";
    AlignedAllocator<double, 32>::deallocate(p, 8);
}

TEST(MemoryAdv, AlignedAlloc_64Byte_Alignment) {
    float* p = AlignedAllocator<float, 64>::allocate(16);
    ASSERT_NE(p, nullptr);
    uintptr_t addr = reinterpret_cast<uintptr_t>(p);
    EXPECT_EQ(addr % 64, 0u);
    AlignedAllocator<float, 64>::deallocate(p, 16);
}

TEST(MemoryAdv, AlignedAlloc_WriteAndRead) {
    // Allocate, write, verify reads back correctly
    double* p = AlignedAllocator<double, 64>::allocate(4);
    ASSERT_NE(p, nullptr);
    p[0] = 1.1; p[1] = 2.2; p[2] = 3.3; p[3] = 4.4;
    EXPECT_DOUBLE_EQ(p[0], 1.1);
    EXPECT_DOUBLE_EQ(p[3], 4.4);
    AlignedAllocator<double, 64>::deallocate(p, 4);
}

TEST(MemoryAdv, AlignedAlloc_MultipleAllocations) {
    std::vector<int*> ptrs;
    for (int i = 0; i < 5; ++i) {
        int* p = AlignedAllocator<int, 32>::allocate(64);
        EXPECT_NE(p, nullptr);
        uintptr_t addr = reinterpret_cast<uintptr_t>(p);
        EXPECT_EQ(addr % 32, 0u);
        ptrs.push_back(p);
    }
    for (int* p : ptrs) AlignedAllocator<int, 32>::deallocate(p, 64);
}

TEST(MemoryAdv, AlignedFreeFunction_Works) {
    void* p = aligned_alloc(64, 256);
    EXPECT_NE(p, nullptr);
    aligned_free(p);
}
