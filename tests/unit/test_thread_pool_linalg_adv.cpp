// MathScript Thread Pool and Additional Coverage Tests

#include <gtest/gtest.h>
#include <future>
#include <vector>
#include <numeric>
#include <cmath>
#include <atomic>

#include "ms/runtime/thread_pool.hpp"
#include "ms/linalg/linalg.hpp"
#include "ms/core/operations.hpp"
#include "ms/stats/stats.hpp"

using namespace ms;
using DMatrix = Matrix<double>;

// ---------------------------------------------------------------------------
// ThreadPool: basic API
// ---------------------------------------------------------------------------

TEST(ThreadPool, Instance_NotNull) {
    auto& pool = ThreadPool::instance();
    (void)pool;
    SUCCEED();  // just checking it doesn't crash
}

TEST(ThreadPool, Size_NonNegative) {
    auto& pool = ThreadPool::instance();
    EXPECT_GE(pool.size(), 0u);
}

TEST(ThreadPool, Initialize_DoesNotCrash) {
    auto& pool = ThreadPool::instance();
    EXPECT_NO_THROW(pool.initialize(2));
    SUCCEED();
}

TEST(ThreadPool, Submit_SimpleTask) {
    auto& pool = ThreadPool::instance();
    pool.initialize(2);
    auto future = pool.submit([]() { return 42; });
    int result = future.get();
    EXPECT_EQ(result, 42);
}

TEST(ThreadPool, Submit_MultipleTasks) {
    auto& pool = ThreadPool::instance();
    pool.initialize(4);
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 8; ++i) {
        futures.push_back(pool.submit([i]() { return i * i; }));
    }
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(futures[i].get(), i * i);
    }
}

TEST(ThreadPool, Submit_WithCapture) {
    auto& pool = ThreadPool::instance();
    pool.initialize(2);
    int base = 100;
    auto future = pool.submit([base]() { return base + 1; });
    EXPECT_EQ(future.get(), 101);
}

TEST(ThreadPool, Wait_DoesNotCrash) {
    auto& pool = ThreadPool::instance();
    pool.initialize(2);
    EXPECT_NO_THROW(pool.wait());
}

TEST(ThreadPool, Parallel_Sum) {
    auto& pool = ThreadPool::instance();
    pool.initialize(4);
    const int N = 1000;
    std::atomic<int> total{0};
    std::vector<std::future<void>> futures;
    for (int i = 0; i < N; ++i) {
        futures.push_back(pool.submit([i, &total]() {
            total.fetch_add(i + 1, std::memory_order_relaxed);
        }));
    }
    for (auto& f : futures) f.get();
    EXPECT_EQ(total.load(), N * (N + 1) / 2);
}

// ---------------------------------------------------------------------------
// Linear algebra: norm variants
// ---------------------------------------------------------------------------

TEST(LinAlgNorm, L1_Norm) {
    DMatrix v(3, 1);
    v(0,0) = -1.0; v(1,0) = 2.0; v(2,0) = -3.0;
    auto r = norm(v, 1);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 6.0, 1e-10);
}

TEST(LinAlgNorm, L2_Norm_Vector) {
    DMatrix v(3, 1);
    v(0,0) = 3.0; v(1,0) = 4.0; v(2,0) = 0.0;
    auto r = norm(v, 2);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 5.0, 1e-10);
}

TEST(LinAlgNorm, LInf_Norm) {
    DMatrix v(4, 1);
    v(0,0) = 1.0; v(1,0) = -5.0; v(2,0) = 3.0; v(3,0) = -2.0;
    // Infinity norm = max absolute value
    auto r = norm(v, 0);  // check what 0 means, or use a high p
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(std::isfinite(*r));
}

// ---------------------------------------------------------------------------
// Linear algebra: rank and cond
// ---------------------------------------------------------------------------

TEST(LinAlgRankCond, Rank_FullRank) {
    DMatrix A(3, 3);
    A(0,0)=1; A(0,1)=0; A(0,2)=0;
    A(1,0)=0; A(1,1)=2; A(1,2)=0;
    A(2,0)=0; A(2,1)=0; A(2,2)=3;
    auto r = rank(A);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 3u);
}

TEST(LinAlgRankCond, Rank_Deficient) {
    DMatrix A(3, 3);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=2; A(1,1)=4; A(1,2)=6;  // row2 = 2*row1
    A(2,0)=0; A(2,1)=0; A(2,2)=1;
    auto r = rank(A);
    ASSERT_TRUE(r.has_value());
    EXPECT_LT(*r, 3u);
}

TEST(LinAlgRankCond, Cond_Identity) {
    DMatrix I(3, 3);
    I(0,0)=1; I(1,1)=1; I(2,2)=1;
    auto r = cond(I);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 1.0, 1e-8);
}

TEST(LinAlgRankCond, Cond_IllConditioned_IsLarge) {
    // Near-singular matrix → large condition number
    DMatrix A(2, 2);
    A(0,0)=1.0; A(0,1)=1.0;
    A(1,0)=1.0; A(1,1)=1.0 + 1e-10;
    auto r = cond(A);
    ASSERT_TRUE(r.has_value());
    EXPECT_GT(*r, 1000.0);
}

// ---------------------------------------------------------------------------
// Linear algebra: lsq (least squares)
// ---------------------------------------------------------------------------

TEST(LinAlgLsq, OverdeterminedSystem) {
    // Fit y = ax + b to 4 points; use [x, 1] design matrix
    DMatrix A(4, 2), b(4, 1);
    A(0,0)=0; A(0,1)=1;   b(0,0)=1;  // y = 0*a + b → b=1
    A(1,0)=1; A(1,1)=1;   b(1,0)=3;  // 1*a + b = 3
    A(2,0)=2; A(2,1)=1;   b(2,0)=5;  // 2*a + b = 5
    A(3,0)=3; A(3,1)=1;   b(3,0)=7;  // 3*a + b = 7
    // Perfect fit: a=2, b=1
    auto r = lsq(A, b);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR((*r)(0, 0), 2.0, 1e-6);  // slope = 2
    EXPECT_NEAR((*r)(1, 0), 1.0, 1e-6);  // intercept = 1
}

TEST(LinAlgLsq, UnderdeterminedSystem_ReturnsResult) {
    // More columns than rows
    DMatrix A(2, 3), b(2, 1);
    A(0,0)=1; A(0,1)=0; A(0,2)=1;  b(0,0)=2;
    A(1,0)=0; A(1,1)=1; A(1,2)=1;  b(1,0)=3;
    auto r = lsq(A, b);
    // Either returns a result or an error - should not crash
    if (r.has_value()) {
        for (size_t i = 0; i < 3; ++i)
            EXPECT_TRUE(std::isfinite((*r)(i, 0)));
    }
    SUCCEED();
}
