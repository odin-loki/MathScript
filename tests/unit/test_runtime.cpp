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

namespace {

long long serial_sum_0_to_n_minus_1(size_t n) {
    long long total = 0;
    for (size_t i = 0; i < n; ++i) {
        total += static_cast<long long>(i);
    }
    return total;
}

} // namespace

TEST(RuntimeTest, parallel_for_n_zero_is_noop) {
    ThreadPool pool;
    std::atomic<int> counter{0};
    pool.parallel_for(0, [&](size_t) { counter.fetch_add(1, std::memory_order_relaxed); });
    EXPECT_EQ(counter.load(), 0);
}

TEST(RuntimeTest, parallel_for_sum_default_grain_matches_serial) {
    ThreadPool pool;
    pool.initialize(4);

    std::atomic<long long> sum{0};
    pool.parallel_for(1000, [&](size_t i) {
        sum.fetch_add(static_cast<long long>(i), std::memory_order_relaxed);
    });

    EXPECT_EQ(sum.load(), serial_sum_0_to_n_minus_1(1000));
    EXPECT_EQ(sum.load(), 499500LL);
}

TEST(RuntimeTest, parallel_for_sum_grain_one_matches_serial) {
    ThreadPool pool;
    pool.initialize(4);

    std::atomic<long long> sum{0};
    pool.parallel_for(1000, 1, [&](size_t i) {
        sum.fetch_add(static_cast<long long>(i), std::memory_order_relaxed);
    });

    EXPECT_EQ(sum.load(), serial_sum_0_to_n_minus_1(1000));
    EXPECT_EQ(sum.load(), 499500LL);
}

TEST(RuntimeTest, parallel_for_sum_large_grain_matches_serial) {
    ThreadPool pool;
    pool.initialize(4);

    std::atomic<long long> sum{0};
    pool.parallel_for(1000, 500, [&](size_t i) {
        sum.fetch_add(static_cast<long long>(i), std::memory_order_relaxed);
    });

    EXPECT_EQ(sum.load(), serial_sum_0_to_n_minus_1(1000));
    EXPECT_EQ(sum.load(), 499500LL);
}

TEST(RuntimeTest, parallel_for_visits_every_index) {
    ThreadPool pool;
    pool.initialize(4);

    std::vector<std::atomic<int>> seen(100);
    for (auto& slot : seen) {
        slot.store(0, std::memory_order_relaxed);
    }

    pool.parallel_for(100, 7, [&](size_t i) {
        seen[i].store(1, std::memory_order_relaxed);
    });

    for (size_t i = 0; i < seen.size(); ++i) {
        EXPECT_EQ(seen[i].load(std::memory_order_relaxed), 1) << "missing index " << i;
    }
}

TEST(RuntimeTest, parallel_for_single_element) {
    ThreadPool pool;
    pool.initialize(2);

    std::atomic<size_t> captured{999};
    pool.parallel_for(1, [&](size_t i) { captured.store(i, std::memory_order_relaxed); });
    EXPECT_EQ(captured.load(), 0u);
}

TEST(RuntimeTest, parallel_for_grain_larger_than_range) {
    ThreadPool pool;
    pool.initialize(2);

    std::atomic<int> counter{0};
    pool.parallel_for(10, 1000, [&](size_t) {
        counter.fetch_add(1, std::memory_order_relaxed);
    });
    EXPECT_EQ(counter.load(), 10);
}

TEST(RuntimeTest, parallel_for_zero_grain_treated_as_one) {
    ThreadPool pool;
    pool.initialize(2);

    std::atomic<int> counter{0};
    pool.parallel_for(5, 0, [&](size_t) {
        counter.fetch_add(1, std::memory_order_relaxed);
    });
    EXPECT_EQ(counter.load(), 5);
}
