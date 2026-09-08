// Regression tests for the audited ms::ThreadPool defects.
//
// The two things asserted here that the old implementation could not satisfy:
//   * parallel_for MOVED the caller's callable into the first chunk
//     (`func = std::forward<F>(f)` inside the chunk loop), so every later chunk
//     captured a moved-from object. With a stateless lambda that is invisible;
//     with any stateful callable the later chunks silently do nothing.
//   * initialize()/wait() were empty bodies and size() was permanently 0, while
//     parallel_for launched ceil(n/grain) simultaneous threads with no bound.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <future>
#include <vector>

#include "ms/runtime/thread_pool.hpp"

using ms::ThreadPool;
using ms::ThreadPoolConfig;

namespace {

// A callable with a DEFINED moved-from state, so "was this functor moved out from
// under a later chunk?" is a deterministic question rather than an
// implementation-defined one.
struct MoveDetectingBody {
    std::vector<std::atomic<int>>* hits;
    std::atomic<int>* calls_through_moved_from;
    bool live = true;

    MoveDetectingBody(std::vector<std::atomic<int>>* h, std::atomic<int>* bad)
        : hits(h), calls_through_moved_from(bad) {}
    MoveDetectingBody(const MoveDetectingBody&) = default;
    MoveDetectingBody& operator=(const MoveDetectingBody&) = default;
    MoveDetectingBody(MoveDetectingBody&& other) noexcept
        : hits(other.hits), calls_through_moved_from(other.calls_through_moved_from),
          live(other.live) {
        other.live = false;
        other.hits = nullptr;
    }
    MoveDetectingBody& operator=(MoveDetectingBody&&) = delete;
    ~MoveDetectingBody() = default;

    void operator()(std::size_t i) const {
        if (!live || hits == nullptr) {
            calls_through_moved_from->fetch_add(1, std::memory_order_relaxed);
            return;
        }
        (*hits)[i].fetch_add(1, std::memory_order_relaxed);
    }
};

} // namespace

TEST(ThreadPoolContract, ParallelForVisitsEveryIndexExactlyOnce) {
    ThreadPool pool;
    pool.initialize(4);

    const std::size_t sizes[] = {1, 2, 3, 5, 7, 10, 17, 64, 100, 1000};
    // grain 0 (must behave as 1), grain not dividing n, grain > n.
    const std::size_t grains[] = {0, 1, 2, 3, 7, 10, 64, 999, 1000, 5000};

    for (std::size_t n : sizes) {
        for (std::size_t grain : grains) {
            std::vector<std::atomic<int>> hits(n);
            for (auto& h : hits) {
                h.store(0, std::memory_order_relaxed);
            }
            std::atomic<int> moved_from_calls{0};

            pool.parallel_for(n, grain, MoveDetectingBody{&hits, &moved_from_calls});

            EXPECT_EQ(moved_from_calls.load(), 0)
                << "n=" << n << " grain=" << grain << ": the callable was moved out from under a chunk";
            for (std::size_t i = 0; i < n; ++i) {
                EXPECT_EQ(hits[i].load(), 1)
                    << "n=" << n << " grain=" << grain << ": index " << i << " visited "
                    << hits[i].load() << " times";
            }
        }
    }
}

TEST(ThreadPoolContract, ParallelForDefaultGrainVisitsEveryIndexExactlyOnce) {
    ThreadPool pool;
    pool.initialize(4);
    for (std::size_t n : {1u, 9u, 100u, 999u}) {
        std::vector<std::atomic<int>> hits(n);
        for (auto& h : hits) {
            h.store(0, std::memory_order_relaxed);
        }
        std::atomic<int> moved_from_calls{0};
        pool.parallel_for(n, MoveDetectingBody{&hits, &moved_from_calls});
        EXPECT_EQ(moved_from_calls.load(), 0) << "n=" << n;
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_EQ(hits[i].load(), 1) << "n=" << n << " index " << i;
        }
    }
}

TEST(ThreadPoolContract, ParallelForNZeroIsANoOp) {
    ThreadPool pool;
    pool.initialize(4);
    std::atomic<int> calls{0};
    pool.parallel_for(0, 1, [&calls](std::size_t) { calls.fetch_add(1); });
    EXPECT_EQ(calls.load(), 0);
}

TEST(ThreadPoolContract, InitializeSetsAndSizeReportsTheConcurrencyWindow) {
    ThreadPool pool;
    EXPECT_EQ(pool.size(), 0u) << "a pool that was never initialised has no window yet";

    ThreadPoolConfig cfg;
    cfg.max_workers = 3;
    pool.configure(cfg);

    pool.initialize(3);
    EXPECT_EQ(pool.size(), 3u);

    // ThreadPoolConfig::max_workers is a real cap, not decoration.
    pool.initialize(64);
    EXPECT_EQ(pool.size(), 3u);

    pool.initialize(0); // 0 means "hardware_concurrency", still capped
    EXPECT_LE(pool.size(), 3u);
    EXPECT_GE(pool.size(), 1u);
}

TEST(ThreadPoolContract, ParallelForNeverExceedsTheConcurrencyWindow) {
    ThreadPool pool;
    ThreadPoolConfig cfg;
    cfg.max_workers = 2;
    pool.configure(cfg);
    pool.initialize(2);
    ASSERT_EQ(pool.size(), 2u);

    std::atomic<int> live{0};
    std::atomic<int> peak{0};
    pool.parallel_for(200, 1, [&live, &peak](std::size_t) {
        const int current = live.fetch_add(1, std::memory_order_acq_rel) + 1;
        int observed = peak.load(std::memory_order_relaxed);
        while (current > observed &&
               !peak.compare_exchange_weak(observed, current, std::memory_order_acq_rel)) {
        }
        for (int spin = 0; spin < 20000; ++spin) {
            std::atomic_signal_fence(std::memory_order_seq_cst);
        }
        live.fetch_sub(1, std::memory_order_acq_rel);
    });
    EXPECT_LE(peak.load(), 2)
        << "parallel_for kept more chunks in flight than the configured window";
    EXPECT_GE(peak.load(), 1);
}

TEST(ThreadPoolContract, MaxQueueSizeBoundsTheChunkCount) {
    ThreadPool pool;
    ThreadPoolConfig cfg;
    cfg.max_workers = 4;
    cfg.max_queue_size = 5; // at most five chunks for the whole range
    pool.configure(cfg);
    pool.initialize(4);

    std::vector<std::atomic<int>> hits(1000);
    for (auto& h : hits) {
        h.store(0, std::memory_order_relaxed);
    }
    std::atomic<int> moved_from_calls{0};
    pool.parallel_for(1000, 1, MoveDetectingBody{&hits, &moved_from_calls});

    EXPECT_EQ(moved_from_calls.load(), 0);
    for (std::size_t i = 0; i < hits.size(); ++i) {
        EXPECT_EQ(hits[i].load(), 1) << "index " << i;
    }
}

TEST(ThreadPoolContract, WaitBlocksUntilEverySubmittedTaskFinished) {
    ThreadPool pool;
    pool.initialize(4);

    std::atomic<int> completed{0};
    std::vector<std::future<void>> futures;
    futures.reserve(16);
    for (int i = 0; i < 16; ++i) {
        futures.push_back(pool.submit([&completed]() {
            for (int spin = 0; spin < 200000; ++spin) {
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }
            completed.fetch_add(1, std::memory_order_relaxed);
        }));
    }

    pool.wait();
    EXPECT_EQ(completed.load(), 16);
    EXPECT_EQ(pool.in_flight(), 0u);

    for (auto& future : futures) {
        future.get();
    }
}

TEST(ThreadPoolContract, SubmitReturnsTaskResults) {
    ThreadPool pool;
    pool.initialize(4);
    std::vector<std::future<int>> futures;
    futures.reserve(8);
    for (int i = 0; i < 8; ++i) {
        futures.push_back(pool.submit([i]() { return i * i; }));
    }
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(futures[static_cast<std::size_t>(i)].get(), i * i);
    }
    pool.wait();
}

TEST(ThreadPoolContract, ConfigRoundTrips) {
    ThreadPool pool;
    ThreadPoolConfig cfg;
    cfg.max_workers = 2;
    cfg.max_queue_size = 7;
    cfg.auto_drain = false;
    pool.configure(cfg);

    const ThreadPoolConfig stored = pool.config();
    EXPECT_EQ(stored.max_workers, 2u);
    EXPECT_EQ(stored.max_queue_size, 7u);
    EXPECT_FALSE(stored.auto_drain);
}
