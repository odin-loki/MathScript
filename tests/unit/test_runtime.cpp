#include <atomic>
#include <future>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "ms/runtime/thread_pool.hpp"

using namespace ms;

TEST(RuntimeTest, thread_pool_submit_and_wait) {
    ThreadPool pool;
    pool.initialize(4);

    constexpr int task_count = 16;
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;
    futures.reserve(task_count);

    for (int i = 0; i < task_count; ++i) {
        futures.push_back(pool.submit([&counter]() { counter.fetch_add(1, std::memory_order_relaxed); }));
    }

    for (auto& future : futures) {
        future.get();
    }

    EXPECT_EQ(counter.load(), task_count);
}

TEST(RuntimeTest, thread_pool_parallel_sum) {
    ThreadPool& pool = ThreadPool::instance();
    pool.initialize();

    constexpr int n = 1000;
    constexpr int chunk_size = 100;
    std::vector<std::future<long long>> partials;

    for (int start = 1; start <= n; start += chunk_size) {
        const int end = (std::min)(start + chunk_size - 1, n);
        partials.push_back(pool.submit([start, end]() {
            long long sum = 0;
            for (int i = start; i <= end; ++i) {
                sum += i;
            }
            return sum;
        }));
    }

    long long total = 0;
    for (auto& future : partials) {
        total += future.get();
    }

    EXPECT_EQ(total, 500500LL);
}

TEST(RuntimeTest, concurrent_tasks_no_crash) {
    ThreadPool pool;
    std::atomic<int> completed{0};
    std::vector<std::future<void>> futures;
    futures.reserve(100);

    for (int i = 0; i < 100; ++i) {
        futures.push_back(pool.submit([&completed, i]() {
            volatile int x = i * i;
            (void)x;
            completed.fetch_add(1, std::memory_order_relaxed);
        }));
    }

    for (auto& future : futures) {
        ASSERT_NO_THROW(future.get());
    }

    EXPECT_EQ(completed.load(), 100);
}

TEST(RuntimeTest, initialize_thread_pool_singleton) {
    ThreadPool& a = initialize_thread_pool();
    ThreadPool& b = ThreadPool::instance();
    EXPECT_EQ(&a, &b);
}
