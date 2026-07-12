// MathScript Runtime Thread Pool Header

#pragma once

#include <algorithm>
#include <future>
#include <thread>
#include <vector>

namespace ms {

// Thread pool configuration
struct ThreadPoolConfig {
    size_t max_workers = std::thread::hardware_concurrency();
    size_t max_queue_size = 1000000;
    bool auto_drain = true;
};

// Task functor
template<typename Func, typename Ret = std::invoke_result_t<Func>>
struct Task {
    Func func;
    Ret result;
    std::future<Ret> future;
};

// Thread pool
class ThreadPool {
public:
    ThreadPool() = default;
    ~ThreadPool();
    
    // Initialize
    void initialize(size_t workers = 0);
    
    // Submit task
    template<typename Func, typename Ret = std::invoke_result_t<Func>>
    std::future<Ret> submit(Func&& func) {
        return std::async(std::launch::async, std::forward<Func>(func));
    }
    
    // Wait for all tasks
    void wait();

    // Parallel for over [0, n) with explicit grain size
    template<typename F>
    void parallel_for(size_t n, size_t grain, F&& f) {
        if (n == 0) {
            return;
        }
        grain = std::max(grain, size_t{1});

        std::vector<std::future<void>> futures;
        futures.reserve((n + grain - 1) / grain);

        for (size_t begin = 0; begin < n; begin += grain) {
            const size_t end = std::min(begin + grain, n);
            futures.push_back(submit([begin, end, func = std::forward<F>(f)]() mutable {
                for (size_t i = begin; i < end; ++i) {
                    func(i);
                }
            }));
        }

        for (auto& future : futures) {
            future.get();
        }
        wait();
    }

    // Parallel for over [0, n) with default grain
    template<typename F>
    void parallel_for(size_t n, F&& f) {
        const size_t workers = std::max(size_t{1}, static_cast<size_t>(std::thread::hardware_concurrency()));
        const size_t grain = std::max(size_t{1}, (n + workers * 4 - 1) / (workers * 4));
        parallel_for(n, grain, std::forward<F>(f));
    }
    
    // Size
    size_t size() const;
    
    // Static instance
    static ThreadPool& instance();
    
private:
    std::vector<std::thread> workers_;
};

// Initialize default thread pool
ThreadPool& initialize_thread_pool();

} // namespace ms