#pragma once

#include "common/ThreadPool.h"
#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct MovableAtomicBool {
    std::atomic<bool> v;
    explicit MovableAtomicBool(bool b = false) noexcept: v(b) {}
    bool load(std::memory_order m = std::memory_order_seq_cst) const noexcept {
        return v.load(m);
    }
    void store(bool b, std::memory_order m = std::memory_order_seq_cst) noexcept {
        v.store(b, m);
    }
    bool exchange(bool b, std::memory_order m = std::memory_order_seq_cst) noexcept {
        return v.exchange(b, m);
    }

    MovableAtomicBool(MovableAtomicBool&& o) noexcept: v(o.v.load(std::memory_order_relaxed)) {}
    MovableAtomicBool& operator=(MovableAtomicBool&& o) noexcept {
        v.store(o.v.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }

    MovableAtomicBool(const MovableAtomicBool&) = delete;
    MovableAtomicBool& operator=(const MovableAtomicBool&) = delete;
};

template<typename T>
class AdaptiveResourcePool {
public:
    struct Params {
        std::function<std::vector<std::unique_ptr<T>>()> resource_initializer;
        std::function<bool(size_t)> can_restore;
        std::function<bool(size_t)> should_release;
        std::function<std::unique_ptr<T>(size_t)> restore_func;
        std::function<void(std::unique_ptr<T>&)> release_func;
        std::shared_ptr<ThreadPool> thread_pool;
        std::function<void(const std::string&)> logger = [](const std::string&) {};
    };

    explicit AdaptiveResourcePool(const Params& params): params_(params) {
        resources_ = params.resource_initializer();
        busy_.resize(resources_.size());
        released_.resize(resources_.size(), false);
    }

    ~AdaptiveResourcePool() {
        for (size_t i = 0; i < resources_.size(); ++i) {
            if (!released_[i]) {
                if (params_.release_func) {
                    params_.release_func(resources_[i]);
                }
                resources_[i].reset();
            }
        }
        resources_.clear();
        busy_.clear();
        released_.clear();
        params_.logger("AdaptiveResourcePool destroyed.");
    }

    // 修改为模板，支持可移动但不可拷贝的任务对象
    template<typename Func>
    void enqueue(Func&& task) {
        maybeRecover();

        for (size_t i = 0; i < resources_.size(); ++i) {
            if (!busy_[i].load() && !released_[i]) {
                if (params_.should_release && params_.should_release(activeCount())) {
                    maybeReleaseOne(i);
                    break;
                }

                busy_[i].store(true);

                // 捕获并移动任务对象，保证可移动lambda能正常工作
                auto task_wrapper = [this, i, task = std::forward<Func>(task)]() mutable {
                    T* res = resources_[i].get();
                    if (!res) {
                        params_.logger("Resource[" + std::to_string(i) + "] is null");
                        busy_[i].store(false);
                        return;
                    }
                    try {
                        task(*res);
                    } catch (const std::exception& e) {
                        params_.logger("Task exception: " + std::string(e.what()));
                    }
                    busy_[i].store(false);
                };

                params_.thread_pool->enqueue(std::move(task_wrapper));
                return;
            }
        }
    }

private:
    size_t activeCount() const {
        return std::count(released_.begin(), released_.end(), false);
    }

    void maybeRecover() {
        size_t active = activeCount();
        if (!params_.can_restore || !params_.can_restore(active))
            return;

        for (size_t i = 0; i < resources_.size(); ++i) {
            if (released_[i]) {
                auto restored = params_.restore_func(i);
                if (restored) {
                    resources_[i] = std::move(restored);
                    released_[i] = false;
                    busy_[i].store(false);
                    params_.logger("Restored resource[" + std::to_string(i) + "]");
                } else {
                    params_.logger("Failed to restore resource[" + std::to_string(i) + "]");
                }
            }
        }
    }

    void maybeReleaseOne(size_t start_index) {
        size_t active = activeCount();
        for (size_t i = start_index; i < resources_.size(); ++i) {
            if (!busy_[i].load() && !released_[i]) {
                if (active <= 1)
                    break;

                busy_[i].store(true);
                params_.release_func(resources_[i]);
                resources_[i].reset();
                released_[i] = true;
                params_.logger("Released resource[" + std::to_string(i) + "]");
                break;
            }
        }
    }

private:
    Params params_;
    std::vector<std::unique_ptr<T>> resources_;
    std::vector<MovableAtomicBool> busy_;
    std::vector<bool> released_;
};
