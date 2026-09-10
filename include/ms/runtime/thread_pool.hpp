// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// MathScript Runtime Thread Pool Header

#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace ms {

// Thread pool configuration.
//
// Every field below is read by ThreadPool; none of them is decorative.
struct ThreadPoolConfig {
    // Upper bound on the number of tasks this pool runs concurrently. initialize()
    // clamps its argument to this value, and parallel_for never keeps more than
    // this many chunks in flight at once.
    size_t max_workers = std::thread::hardware_concurrency();
    // Upper bound on the number of chunks parallel_for is allowed to create. When
    // (n, grain) would produce more, the grain is enlarged until it does not.
    size_t max_queue_size = 1000000;
    // When true, parallel_for calls wait() before returning.
    bool auto_drain = true;
};

// Task functor
template<typename Func, typename Ret = std::invoke_result_t<Func>>
struct Task {
    Func func;
    Ret result;
    std::future<Ret> future;
};

// Asynchronous task runner with a bounded concurrency window.
//
// HONEST DESCRIPTION OF THE EXECUTION MODEL: this class does not own persistent
// worker threads. Each submit() launches its callable with
// std::async(std::launch::async), i.e. on a fresh thread, and returns that task's
// future. What the class does provide, and what initialize()/size()/wait() and
// ThreadPoolConfig actually mean:
//
//   * initialize(workers) sets the concurrency window (clamped to
//     ThreadPoolConfig::max_workers, minimum 1); size() reports it.
//   * parallel_for keeps at most `size()` chunks in flight at a time, so a call
//     such as parallel_for(1000, 1, f) runs in waves of size() tasks rather than
//     launching 1000 threads at once.
//   * wait() blocks until every task submitted through this object has finished
//     executing.
//   * The destructor waits for outstanding tasks, because each running task holds
//     a reference back into the pool's in-flight counter.
//
// Nothing here reuses a thread across tasks. If per-task thread creation becomes a
// measured cost, the replacement is a real worker-queue pool; until then this
// documentation is the contract.
class ThreadPool {
public:
    ThreadPool() = default;
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    // Replace the configuration. Takes effect on the next initialize() and on
    // every subsequent parallel_for.
    void configure(const ThreadPoolConfig& config);
    ThreadPoolConfig config() const;

    // Set the concurrency window. workers == 0 means hardware_concurrency().
    // The result is clamped to [1, ThreadPoolConfig::max_workers] and is what
    // size() returns afterwards.
    void initialize(size_t workers = 0);

    // Submit task. Runs on a new thread; the returned future is the only handle.
    template<typename Func, typename Ret = std::invoke_result_t<Func>>
    std::future<Ret> submit(Func&& func) {
        task_started();
        return std::async(
            std::launch::async,
            [this, fn = std::forward<Func>(func)]() mutable -> Ret {
                InFlightGuard guard(*this);
                return fn();
            });
    }

    // Wait for all tasks submitted through this pool to finish executing.
    void wait();

    // Parallel for over [0, n) with explicit grain size
    template<typename F>
    void parallel_for(size_t n, size_t grain, F&& f) {
        if (n == 0) {
            return;
        }
        grain = std::max(grain, size_t{1});

        // Keep the chunk count inside ThreadPoolConfig::max_queue_size.
        const size_t chunk_cap = std::max(chunk_limit(), size_t{1});
        if ((n + grain - 1) / grain > chunk_cap) {
            grain = (n + chunk_cap - 1) / chunk_cap;
        }

        const size_t width = concurrency();
        if (width <= 1 || grain >= n) {
            for (size_t i = 0; i < n; ++i) {
                f(i);
            }
            return;
        }

        std::vector<std::future<void>> wave;
        wave.reserve(width);

        for (size_t begin = 0; begin < n; begin += grain) {
            const size_t end = std::min(begin + grain, n);
            // f is REFERENCED, never moved. `func = std::forward<F>(f)` inside this
            // loop moves the callable into the first chunk and leaves every later
            // chunk capturing a moved-from object.
            wave.push_back(submit([begin, end, fn = std::ref(f)]() {
                for (size_t i = begin; i < end; ++i) {
                    fn(i);
                }
            }));
            if (wave.size() >= width) {
                for (auto& future : wave) {
                    future.get();
                }
                wave.clear();
            }
        }

        for (auto& future : wave) {
            future.get();
        }
        if (drains_automatically()) {
            wait();
        }
    }

    // Parallel for over [0, n) with default grain
    template<typename F>
    void parallel_for(size_t n, F&& f) {
        const size_t workers = std::max(size_t{1}, concurrency());
        const size_t grain = std::max(size_t{1}, (n + workers * 4 - 1) / (workers * 4));
        parallel_for(n, grain, std::forward<F>(f));
    }

    // Number of tasks this pool runs concurrently.
    size_t size() const;

    // Tasks submitted through this pool that have not finished executing.
    size_t in_flight() const;

    // Static instance
    static ThreadPool& instance();

private:
    void task_started();
    void task_finished();

    // Concurrency window, initialising it on first use when the caller never
    // called initialize().
    size_t concurrency();
    size_t chunk_limit() const;
    bool drains_automatically() const;

    struct InFlightGuard {
        explicit InFlightGuard(ThreadPool& pool) : pool_(pool) {}
        InFlightGuard(const InFlightGuard&) = delete;
        InFlightGuard& operator=(const InFlightGuard&) = delete;
        ~InFlightGuard() { pool_.task_finished(); }
        ThreadPool& pool_;
    };

    mutable std::mutex mutex_;
    std::condition_variable idle_cv_;
    ThreadPoolConfig config_{};
    size_t concurrency_ = 0;
    size_t in_flight_ = 0;
};

// Initialize default thread pool
ThreadPool& initialize_thread_pool();

} // namespace ms
