// MathScript Memory Allocators Advanced Unit Tests
// Tests: AlignedAllocator, PoolAllocator, PinnedAllocator usage patterns

#include <gtest/gtest.h>
#include <vector>
#include <cstddef>
#include <cstring>
#include <memory>
#include <type_traits>

#include "ms/memory/aligned_allocator.hpp"
#include "ms/memory/pool_allocator.hpp"
#include "ms/memory/pinned_allocator.hpp"

using namespace ms::memory;

// ---------------------------------------------------------------------------
// AlignedAllocator
// ---------------------------------------------------------------------------

TEST(AlignedAllocatorTest, Allocate_Returns_NonNull) {
    AlignedAllocator<double, 64> alloc;
    double* p = alloc.allocate(16);
    ASSERT_NE(p, nullptr);
    alloc.deallocate(p, 16);
}

TEST(AlignedAllocatorTest, Allocation_Is_Aligned) {
    AlignedAllocator<double, 64> alloc;
    double* p = alloc.allocate(8);
    ASSERT_NE(p, nullptr);
    // Check 64-byte alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 64, 0u);
    alloc.deallocate(p, 8);
}

TEST(AlignedAllocatorTest, WritableMemory) {
    AlignedAllocator<float, 32> alloc;
    float* p = alloc.allocate(8);
    ASSERT_NE(p, nullptr);
    for (int i = 0; i < 8; ++i) p[i] = static_cast<float>(i);
    for (int i = 0; i < 8; ++i) EXPECT_NEAR(p[i], static_cast<float>(i), 1e-6f);
    alloc.deallocate(p, 8);
}

TEST(AlignedAllocatorTest, DefaultAlignment_AtLeastAlignof) {
    AlignedAllocator<double> alloc;
    double* p = alloc.allocate(4);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % alignof(double), 0u);
    alloc.deallocate(p, 4);
}

TEST(AlignedAllocatorTest, RawArray_Writable) {
    AlignedAllocator<int, 64> alloc;
    int* p = alloc.allocate(100);
    ASSERT_NE(p, nullptr);
    for (int i = 0; i < 100; ++i) p[i] = i * 2;
    EXPECT_EQ(p[50], 100);
    EXPECT_EQ(p[99], 198);
    alloc.deallocate(p, 100);
}

TEST(AlignedAllocatorTest, LargeAllocation) {
    AlignedAllocator<double, 64> alloc;
    const size_t N = 1024;
    double* p = alloc.allocate(N);
    ASSERT_NE(p, nullptr);
    std::memset(p, 0, N * sizeof(double));
    EXPECT_NEAR(p[0], 0.0, 1e-12);
    EXPECT_NEAR(p[N-1], 0.0, 1e-12);
    alloc.deallocate(p, N);
}

// ---------------------------------------------------------------------------
// PoolAllocator<BlockSize> — fixed block-size pool
// ---------------------------------------------------------------------------

TEST(PoolAllocatorTest, BasicAllocate_32bytes) {
    PoolAllocator<32> pool;
    void* p = pool.allocate();
    ASSERT_NE(p, nullptr);
    // Write a value to verify the memory is accessible
    *static_cast<int*>(p) = 42;
    EXPECT_EQ(*static_cast<int*>(p), 42);
    pool.deallocate(p);
}

TEST(PoolAllocatorTest, MultipleAllocations_64bytes) {
    PoolAllocator<64> pool;
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        void* p = pool.allocate();
        ASSERT_NE(p, nullptr);
        *static_cast<int*>(p) = i;
        ptrs.push_back(p);
    }
    for (int i = 0; i < 10; ++i)
        EXPECT_EQ(*static_cast<int*>(ptrs[i]), i);
    for (auto p : ptrs) pool.deallocate(p);
}

TEST(PoolAllocatorTest, Deallocate_And_Reallocate) {
    PoolAllocator<16> pool;
    void* p1 = pool.allocate();
    pool.deallocate(p1);
    void* p2 = pool.allocate();
    ASSERT_NE(p2, nullptr);
    pool.deallocate(p2);
}

TEST(PoolAllocatorTest, Allocated_Count_Increases) {
    PoolAllocator<32> pool;
    EXPECT_EQ(pool.allocated(), 0u);
    void* p = pool.allocate();
    EXPECT_EQ(pool.allocated(), 1u);
    pool.deallocate(p);
    EXPECT_EQ(pool.allocated(), 0u);
}

TEST(PoolAllocatorTest, HighCapacity) {
    PoolAllocator<32> pool;
    std::vector<void*> ptrs;
    // Allocate more than one slab worth to test grow()
    for (int i = 0; i < 100; ++i)
        ptrs.push_back(pool.allocate());
    EXPECT_EQ(pool.allocated(), 100u);
    for (auto p : ptrs) pool.deallocate(p);
    EXPECT_EQ(pool.allocated(), 0u);
}

// ---------------------------------------------------------------------------
// PinnedAllocator (CPU fallback: wraps malloc)
// PinnedAllocator is not templated - it allocates char* arrays
// ---------------------------------------------------------------------------

TEST(PinnedAllocatorTest, Allocate_NonNull) {
    PinnedAllocator alloc;
    char** p = alloc.allocate(16);
    ASSERT_NE(p, nullptr);
    alloc.deallocate(p, 16);
}

TEST(PinnedAllocatorTest, Allocate_Writable) {
    PinnedAllocator alloc;
    char** p = alloc.allocate(8);
    ASSERT_NE(p, nullptr);
    // Use as raw storage
    char* bytes = reinterpret_cast<char*>(p);
    bytes[0] = 'X';
    EXPECT_EQ(bytes[0], 'X');
    alloc.deallocate(p, 8);
}

TEST(PinnedAllocatorTest, MultipleAllocations) {
    PinnedAllocator alloc;
    std::vector<char**> ptrs;
    for (int i = 0; i < 5; ++i)
        ptrs.push_back(alloc.allocate(4));
    for (auto p : ptrs) {
        ASSERT_NE(p, nullptr);
        alloc.deallocate(p, 4);
    }
}

// ---------------------------------------------------------------------------
// aligned_alloc / aligned_free free functions
// ---------------------------------------------------------------------------

TEST(AlignedFunctions, AlignedAlloc_Returns_NonNull) {
    void* p = aligned_alloc(64, 256);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 64, 0u);
    aligned_free(p);
}

TEST(AlignedFunctions, AlignedAlloc_16byte_Alignment) {
    void* p = aligned_alloc(16, 128);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 16, 0u);
    aligned_free(p);
}
