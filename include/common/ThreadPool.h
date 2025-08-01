// Copyright 2025 Xiaojian Wu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <vector>

#include <boost/lockfree/queue.hpp>
#include <boost/lockfree/stack.hpp>

/**
 * @brief Highly optimized thread pool:
 *   - Two lock-free MPMC queues (high/normal)
 *   - Object pool for TaskItem nodes (no per-task new/delete)
 *   - Spin-then-block wait strategy
 *   - Cache-line–padded counters to avoid false sharing
 *   - Per-task timeout via std::jthread
 */
class ThreadPool {
public:
    explicit ThreadPool(
        size_t num_threads,
        size_t max_pending = 100,
        size_t queue_capacity = 1024,
        size_t pool_capacity = 1024
    ):
        max_pending_(max_pending),
        high_q_(queue_capacity),
        normal_q_(queue_capacity),
        pool_(pool_capacity) {
        // Pre-allocate TaskItem nodes
        for (size_t i = 0; i < pool_capacity; ++i)
            pool_.push(new TaskItem);

        // Launch workers
        for (size_t i = 0; i < num_threads; ++i)
            workers_.emplace_back([this](std::stop_token st) { workerLoop(st); });
    }

    ~ThreadPool() {
        stop_.store(true, std::memory_order_relaxed);
        cv_.notify_all();
        // workers (std::jthread) join automatically
        // cleanup pool
        TaskItem* item;
        while (pool_.pop(item))
            delete item;
    }

    /// Enqueue a task, optionally high-priority or with timeout_ms
    template<class F>
    std::future<void> enqueue(F&& fn, int timeout_ms = -1, bool high_priority = false) {
        auto prom = std::make_shared<std::promise<void>>();
        auto fut = prom->get_future();
        std::stop_source stop_src;

        // Fetch node from pool
        TaskItem* node = nullptr;
        if (!pool_.pop(node)) {
            node = new TaskItem; // fallback
        }

        // Bind user function
        node->timeout_ms = timeout_ms;
        node->stop_src = stop_src;
        node->func = [fn = std::forward<F>(fn), prom](std::stop_token tok) mutable {
            try {
                if constexpr (std::is_invocable_v<F, std::stop_token>)
                    fn(tok);
                else
                    fn();
                prom->set_value();
            } catch (...) {
                prom->set_exception(std::current_exception());
            }
        };

        // Overload protection
        size_t prev = pending_.fetch_add(1, std::memory_order_relaxed);
        if (prev >= max_pending_) {
            TaskItem* dropped = nullptr;
            normal_q_.pop(dropped);
            if (dropped) {
                std::cerr << "[ThreadPool] Warning: Dropped oldest task\n";
                cleanupNode(dropped);
                pending_.fetch_sub(1, std::memory_order_relaxed);
            }
        }

        // Enqueue to appropriate queue
        if (high_priority)
            high_q_.push(node);
        else
            normal_q_.push(node);

        cv_.notify_one();
        return fut;
    }

    /// Number of tasks waiting or running
    size_t pendingTasks() const {
        return pending_.load(std::memory_order_relaxed);
    }

    /// Block until all tasks complete
    void waitUntilEmpty() {
        std::unique_lock<std::mutex> lk(done_mtx_);
        done_cv_.wait(lk, [this] {
            return pending_.load(std::memory_order_relaxed) == 0
                && busy_.load(std::memory_order_relaxed) == 0;
        });
    }

private:
    struct TaskItem {
        std::function<void(std::stop_token)> func;
        std::stop_source stop_src;
        int timeout_ms;
    };

    // Return node to pool
    void cleanupNode(TaskItem* node) {
        // clear functor to free captures
        node->func = nullptr;
        pool_.push(node);
    }

    void workerLoop(std::stop_token pool_stop) {
        while (!pool_stop.stop_requested() && !stop_.load(std::memory_order_relaxed)) {
            TaskItem* item = nullptr;
            // spin-then-block:
            for (int i = 0; i < 100; ++i) {
                if (high_q_.pop(item) || normal_q_.pop(item))
                    break;
                __asm__ volatile("pause");
            }
            if (!item) {
                std::unique_lock<std::mutex> lk(wait_mtx_);
                cv_.wait(lk, [this] {
                    return stop_.load(std::memory_order_relaxed) || !high_q_.empty()
                        || !normal_q_.empty();
                });
                // try once more
                high_q_.pop(item) || normal_q_.pop(item);
            }

            if (!item)
                continue; // spurious wake

            busy_.fetch_add(1, std::memory_order_relaxed);

            // timeout thread
            std::jthread timer;
            if (item->timeout_ms > 0) {
                timer = std::jthread([&, ms = item->timeout_ms](std::stop_token t) {
                    if (!t.stop_requested())
                        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
                    if (!t.stop_requested())
                        item->stop_src.request_stop();
                });
            }

            // execute
            item->func(item->stop_src.get_token());
            if (timer.joinable())
                timer.request_stop();

            // cleanup
            busy_.fetch_sub(1, std::memory_order_relaxed);
            pending_.fetch_sub(1, std::memory_order_relaxed);
            if (pending_.load() == 0 && busy_.load() == 0) {
                std::lock_guard<std::mutex> lk(done_mtx_);
                done_cv_.notify_all();
            }
            cleanupNode(item);
        }
    }

    // Workers
    std::vector<std::jthread> workers_;
    std::atomic<bool> stop_ { false };

    // Two lock-free queues
    boost::lockfree::queue<TaskItem*> high_q_;
    boost::lockfree::queue<TaskItem*> normal_q_;

    // Object pool for TaskItem nodes
    boost::lockfree::stack<TaskItem*> pool_;

    // Counters (cache-line padded)
    alignas(64) std::atomic<size_t> pending_ { 0 };
    alignas(64) std::atomic<size_t> busy_ { 0 };

    // Config
    size_t max_pending_;

    // Waiting
    std::mutex wait_mtx_;
    std::condition_variable cv_;

    // Completion notification
    std::mutex done_mtx_;
    std::condition_variable done_cv_;
};
