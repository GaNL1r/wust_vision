#pragma once
#include <functional>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

struct TbbThreadPool {
    tbb::task_arena arena;
    TbbThreadPool(int threads = std::thread::hardware_concurrency()): arena(threads) {}

    void enqueue(std::function<void()> task) {
        arena.enqueue([task = std::move(task)] { task(); });
    }
};
