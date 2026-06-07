#include "ms/runtime/thread_pool.hpp"

namespace ms {

ThreadPool::~ThreadPool() {
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::initialize(size_t) {}

void ThreadPool::wait() {}

size_t ThreadPool::size() const {
    return workers_.size();
}

ThreadPool& ThreadPool::instance() {
    static ThreadPool pool;
    return pool;
}

ThreadPool& initialize_thread_pool() {
    return ThreadPool::instance();
}

} // namespace ms
