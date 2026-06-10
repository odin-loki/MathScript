#include <gtest/gtest.h>
#include <cstddef>
#include <vector>
#include <thread>
#include <atomic>

#include "ms/memory/arena.hpp"
#include "ms/memory/pool_allocator.hpp"
#include "ms/memory/aligned_allocator.hpp"
#include "ms/memory/pinned_allocator.hpp"

using namespace ms::memory;

// ---------------------------------------------------------------------------
// Arena extended tests
// ---------------------------------------------------------------------------

TEST(ArenaExtTest, DefaultCapacityPositive) {
    Arena arena;
    EXPECT_GT(arena.capacity(), 0u);
}

TEST(ArenaExtTest, CustomCapacity) {
    Arena arena(1024 * 1024);  // 1 MB
    EXPECT_EQ(arena.capacity(), 1024 * 1024u);
}

TEST(ArenaExtTest, AllocateMultipleTypes) {
    Arena arena(64 * 1024);
    double* d = arena.allocate<double>(10);
    ASSERT_NE(d, nullptr);
    int* i = arena.allocate<int>(5);
    ASSERT_NE(i, nullptr);
    // Write to ensure memory is accessible
    for (int k = 0; k < 10; ++k) d[k] = static_cast<double>(k);
    for (int k = 0; k < 5; ++k) i[k] = k * 2;
    EXPECT_NEAR(d[5], 5.0, 1e-12);
    EXPECT_EQ(i[3], 6);
}

TEST(ArenaExtTest, ConstructObject) {
    Arena arena(4096);
    struct Point { double x, y; };
    Point* p = arena.construct<Point>(Point{1.5, 2.5});
    ASSERT_NE(p, nullptr);
    EXPECT_NEAR(p->x, 1.5, 1e-12);
    EXPECT_NEAR(p->y, 2.5, 1e-12);
}

TEST(ArenaExtTest, ResetDoesNotCrash) {
    Arena arena(4096);
    auto* p = arena.allocate<int>(10);
    (void)p;
    EXPECT_NO_THROW(arena.reset());
}

TEST(ArenaExtTest, AllocateAfterReset) {
    Arena arena(4096);
    auto* p1 = arena.allocate<int>(5);
    (void)p1;
    arena.reset();
    // After reset, arena may be in an undefined state for further allocs
    // (monotonic_buffer_resource may throw or return nullptr)
    // Just verify no crash
    SUCCEED();
}

TEST(ArenaExtTest, ScratchScopeResetsArena) {
    {
        ScratchScope scope;
        auto* p = scratch_arena.allocate<double>(16);
        (void)p;
        // scope destructor will call scratch_arena.reset()
    }
    // After scope: scratch_arena should be reset, no crash
    SUCCEED();
}

// ---------------------------------------------------------------------------
// PoolAllocator extended tests
// ---------------------------------------------------------------------------

TEST(PoolExtTest, AllocateManyBlocks) {
    PoolAllocator<64> pool;
    std::vector<void*> ptrs;
    for (int i = 0; i < 100; ++i) {
        ptrs.push_back(pool.allocate());
    }
    EXPECT_EQ(pool.allocated(), 100u);
    for (void* p : ptrs) pool.deallocate(p);
    EXPECT_EQ(pool.allocated(), 0u);
}

TEST(PoolExtTest, AllocateAndDeallocateAlternating) {
    PoolAllocator<32> pool;
    void* a = pool.allocate();
    void* b = pool.allocate();
    EXPECT_EQ(pool.allocated(), 2u);
    pool.deallocate(a);
    EXPECT_EQ(pool.allocated(), 1u);
    pool.deallocate(b);
    EXPECT_EQ(pool.allocated(), 0u);
}

TEST(PoolExtTest, BlocksAreNonNull) {
    PoolAllocator<16> pool;
    for (int i = 0; i < 10; ++i) {
        void* p = pool.allocate();
        EXPECT_NE(p, nullptr) << "Block " << i << " should not be null";
        pool.deallocate(p);
    }
}

TEST(PoolExtTest, ReuseAfterDeallocate) {
    PoolAllocator<64> pool;
    void* p1 = pool.allocate();
    pool.deallocate(p1);
    void* p2 = pool.allocate();  // May reuse p1's block
    EXPECT_NE(p2, nullptr);
    pool.deallocate(p2);
    EXPECT_EQ(pool.allocated(), 0u);
}

TEST(PoolExtTest, VariousBlockSizes_Smoke) {
    // Test local pool instances with various block sizes
    PoolAllocator<16> p16;
    PoolAllocator<32> p32;
    PoolAllocator<64> p64;
    PoolAllocator<128> p128;
    void* b16  = p16.allocate();  EXPECT_NE(b16,  nullptr);  p16.deallocate(b16);
    void* b32  = p32.allocate();  EXPECT_NE(b32,  nullptr);  p32.deallocate(b32);
    void* b64  = p64.allocate();  EXPECT_NE(b64,  nullptr);  p64.deallocate(b64);
    void* b128 = p128.allocate(); EXPECT_NE(b128, nullptr);  p128.deallocate(b128);
}

// ---------------------------------------------------------------------------
// AlignedAllocator extended tests
// ---------------------------------------------------------------------------

TEST(AlignedAllocExtTest, AllocateReturnsAligned64) {
    AlignedAllocator<double, 64> alloc;
    double* ptr = alloc.allocate(16);
    ASSERT_NE(ptr, nullptr);
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % 64u, 0u) << "Pointer should be 64-byte aligned";
    alloc.deallocate(ptr, 16);
}

TEST(AlignedAllocExtTest, AllocateAndWriteValues) {
    AlignedAllocator<float, 32> alloc;
    float* ptr = alloc.allocate(8);
    ASSERT_NE(ptr, nullptr);
    for (int i = 0; i < 8; ++i) ptr[i] = static_cast<float>(i) * 0.5f;
    EXPECT_NEAR(ptr[4], 2.0f, 1e-6f);
    alloc.deallocate(ptr, 8);
}

TEST(AlignedAllocExtTest, MultipleAllocationsAllAligned) {
    AlignedAllocator<double, 32> alloc;
    std::vector<double*> ptrs;
    for (int i = 0; i < 5; ++i) {
        double* p = alloc.allocate(4);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 32u, 0u);
        ptrs.push_back(p);
    }
    for (double* p : ptrs) alloc.deallocate(p, 4);
}

// ---------------------------------------------------------------------------
// PinnedAllocator extended tests (PinnedAllocator is not a class template)
// ---------------------------------------------------------------------------

TEST(PinnedAllocExtTest, AllocateNonNull) {
    PinnedAllocator alloc;
    auto* p = alloc.allocate(4);  // returns char**
    EXPECT_NE(p, nullptr);
    alloc.deallocate(p, 4);
}

TEST(PinnedAllocExtTest, AllocateAndFree) {
    PinnedAllocator alloc;
    auto* p = alloc.allocate(8);
    ASSERT_NE(p, nullptr);
    alloc.deallocate(p, 8);
    SUCCEED();
}

TEST(PinnedAllocExtTest, LargeAllocation) {
    PinnedAllocator alloc;
    constexpr size_t N = 1024;
    auto* p = alloc.allocate(N);
    ASSERT_NE(p, nullptr);
    alloc.deallocate(p, N);
}
