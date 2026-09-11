// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "ms/runtime/thread_pool.hpp"

namespace ms {

ThreadPool::~ThreadPool() {
    // Every running task holds an InFlightGuard referring back to this object, so
    // the pool must outlive them.
    wait();
}

void ThreadPool::configure(const ThreadPoolConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    if (concurrency_ != 0) {
        const size_t cap = (config_.max_workers == 0) ? size_t{1} : config_.max_workers;
        concurrency_ = std::min(concurrency_, cap);
    }
}

ThreadPoolConfig ThreadPool::config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

void ThreadPool::initialize(size_t workers) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t target = workers;
    if (target == 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        target = (hw == 0) ? size_t{1} : static_cast<size_t>(hw);
    }
    const size_t cap = (config_.max_workers == 0) ? size_t{1} : config_.max_workers;
    target = std::min(target, cap);
    concurrency_ = std::max(target, size_t{1});
}

void ThreadPool::task_started() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++in_flight_;
}

void ThreadPool::task_finished() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (in_flight_ > 0) {
            --in_flight_;
        }
    }
    idle_cv_.notify_all();
}

void ThreadPool::wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    idle_cv_.wait(lock, [this] { return in_flight_ == 0; });
}

size_t ThreadPool::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return concurrency_;
}

size_t ThreadPool::in_flight() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return in_flight_;
}

size_t ThreadPool::concurrency() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (concurrency_ != 0) {
            return concurrency_;
        }
    }
    initialize(0);
    std::lock_guard<std::mutex> lock(mutex_);
    return concurrency_;
}

size_t ThreadPool::chunk_limit() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.max_queue_size;
}

bool ThreadPool::drains_automatically() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.auto_drain;
}

ThreadPool& ThreadPool::instance() {
    static ThreadPool pool;
    return pool;
}

ThreadPool& initialize_thread_pool() {
    return ThreadPool::instance();
}

} // namespace ms
