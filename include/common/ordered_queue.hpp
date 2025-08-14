#pragma once
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <map>
#include <mutex>

template<typename T>
class OrderedQueue {
public:
    static_assert(
        std::is_same<decltype(T::id), int>::value
            && std::is_same<decltype(T::timestamp), std::chrono::steady_clock::time_point>::value,
        "T must have int id and timestamp of type std::chrono::steady_clock::time_point"
    );

    OrderedQueue(int max_wait_ms = 50): current_id_(0), max_wait_ms_(max_wait_ms) {}

    // 入队
    void enqueue(const T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (item.id < current_id_) {
            // 旧 ID 帧，缓存到 old_buffer_ 以保持顺序
            old_buffer_[item.id] = item;
        } else {
            buffer_[item.id] = item;
        }
        cond_var_.notify_one();
    }

    // 非阻塞尝试出队
    bool try_dequeue(T& out_item) {
        std::lock_guard<std::mutex> lock(mutex_);

        // 先输出 old_buffer 中比 current_id_ 小的帧
        if (!old_buffer_.empty()) {
            auto it_old = old_buffer_.begin();
            out_item = it_old->second;
            old_buffer_.erase(it_old);
            return true;
        }

        if (buffer_.empty())
            return false;

        auto it = buffer_.begin();
        auto now = std::chrono::steady_clock::now();

        // 如果缺失 ID 超时，则跳帧
        if (it->first > current_id_ + 1) {
            auto wait_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second.timestamp)
                    .count();
            if (wait_duration >= max_wait_ms_) {
                current_id_ = it->first;
            } else {
                return false; // 等待缺失帧
            }
        }

        out_item = it->second;
        current_id_ = it->first;
        buffer_.erase(it);
        return true;
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.size() + old_buffer_.size();
    }

private:
    std::map<int, T> buffer_; // 按 ID 排序的主缓冲
    std::map<int, T> old_buffer_; // 小于 current_id_ 的旧 ID 帧缓存
    int current_id_;
    int max_wait_ms_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
};
