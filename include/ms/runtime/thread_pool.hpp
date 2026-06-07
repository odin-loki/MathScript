// MathScript Runtime Thread Pool Header

#pragma once

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