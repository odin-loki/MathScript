#include <gtest/gtest.h>

#include "ms/memory/aligned_allocator.hpp"
#include "ms/memory/arena.hpp"
#include "ms/memory/pinned_allocator.hpp"
#include "ms/memory/policy.hpp"
#include "ms/memory/pool_allocator.hpp"

using namespace ms::memory;

TEST(MemoryTest, arena_bytes_used_empty) {
    Arena arena(4096);
    EXPECT_EQ(arena.bytes_used(), 0u);
}

TEST(MemoryTest, arena_bytes_used_after_single_allocation) {
    Arena arena(4096);
    (void)arena.allocate<int>();
    EXPECT_EQ(arena.bytes_used(), sizeof(int));
}

TEST(MemoryTest, arena_bytes_used_cumulative) {
    Arena arena(4096);
    (void)arena.allocate<int>(4);
    const size_t after_ints = arena.bytes_used();
    EXPECT_EQ(after_ints, 4u * sizeof(int));

    (void)arena.allocate<double>(2);
    EXPECT_EQ(arena.bytes_used(), after_ints + 2u * sizeof(double));
}

TEST(MemoryTest, arena_bytes_used_construct_counts) {
    Arena arena(4096);
    (void)arena.construct<int>(42);
    EXPECT_EQ(arena.bytes_used(), sizeof(int));
}

TEST(MemoryTest, arena_bytes_used_reset_clears) {
    Arena arena(4096);
    (void)arena.allocate<int>(8);
    (void)arena.allocate<double>(4);
    EXPECT_GT(arena.bytes_used(), 0u);

    arena.reset();
    EXPECT_EQ(arena.bytes_used(), 0u);
}

TEST(MemoryTest, arena_bytes_used_after_reset_only_new) {
    Arena arena(4096);
    (void)arena.allocate<int>(10);
    arena.reset();

    (void)arena.allocate<char>(3);
    EXPECT_EQ(arena.bytes_used(), 3u * sizeof(char));
}

TEST(MemoryTest, arena_bytes_used_multiple_types) {
    Arena arena(8192);
    (void)arena.allocate<int>(1);
    (void)arena.allocate<double>(1);
    (void)arena.allocate<char>(16);
    EXPECT_EQ(arena.bytes_used(),
              sizeof(int) + sizeof(double) + 16u * sizeof(char));
}

TEST(MemoryTest, arena_allocate_and_reset) {
    Arena arena(4096);
    int* a = arena.construct<int>(1);
    int* b = arena.allocate<int>(2);
    b[0] = 2;
    b[1] = 3;
    EXPECT_EQ(*a, 1);
    EXPECT_EQ(b[0], 2);
    EXPECT_EQ(b[1], 3);

    arena.reset();

    double* c = arena.construct<double>(3.14);
    EXPECT_DOUBLE_EQ(*c, 3.14);

    {
        ScratchScope scope;
        int* scratch = scratch_arena.construct<int>(7);
        EXPECT_EQ(*scratch, 7);
    }
}

TEST(MemoryTest, pool_reuse_blocks) {
    PoolAllocator<64> pool;
    void* first = pool.allocate();
    ASSERT_NE(first, nullptr);
    pool.deallocate(first);

    void* second = pool.allocate();
    EXPECT_EQ(first, second);
}

TEST(MemoryTest, allocation_size_tracked) {
    PoolAllocator<32> pool;
    EXPECT_EQ(pool.allocated(), 0u);

    void* a = pool.allocate();
    void* b = pool.allocate();
    void* c = pool.allocate();
    EXPECT_EQ(pool.allocated(), 3u);

    pool.deallocate(b);
    EXPECT_EQ(pool.allocated(), 2u);

    pool.deallocate(a);
    pool.deallocate(c);
    EXPECT_EQ(pool.allocated(), 0u);
}

TEST(MemoryTest, arena_overflow_handled) {
    Arena arena(128);
    std::vector<int*> pointers;
    pointers.reserve(64);

    for (int i = 0; i < 64; ++i) {
        int* ptr = arena.construct<int>(i);
        ASSERT_NE(ptr, nullptr);
        EXPECT_EQ(*ptr, i);
        pointers.push_back(ptr);
    }

    arena.reset();
    int* after_reset = arena.construct<int>(99);
    ASSERT_NE(after_reset, nullptr);
    EXPECT_EQ(*after_reset, 99);
}

TEST(MemoryTest, aligned_allocator_smoke) {
    double* data = AlignedAllocator<double, 64>::allocate(8);
    ASSERT_NE(data, nullptr);
    for (size_t i = 0; i < 8; ++i) {
        data[i] = static_cast<double>(i);
    }
    EXPECT_DOUBLE_EQ(data[3], 3.0);
    AlignedAllocator<double, 64>::deallocate(data, 8);
}

TEST(MemoryTest, aligned_size_policy) {
    EXPECT_EQ(aligned_size(1, 64), 64u);
    EXPECT_EQ(aligned_size(64, 64), 64u);
    EXPECT_EQ(aligned_size(65, 64), 128u);
}

TEST(MemoryTest, pinned_allocator_alloc_free) {
    PinnedAllocator::value_type* data = PinnedAllocator::allocate(4);
    ASSERT_NE(data, nullptr);
    PinnedAllocator::deallocate(data, 4);
}
