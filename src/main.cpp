
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
#include "wust_vision.hpp"
#include <csignal>
#include <mutex>
#include <string>
#include <unistd.h>

std::mutex mtx;
std::condition_variable c;

void signalHandler(int signum) {
    WUST_MAIN("main") << "Interrupt signal (" << signum << ") received.";
    gobal::exit_flag.store(true, std::memory_order_release);
}

int main() {
    WustVision vision;
    if (vision.init()) {
        vision.run();
    }

    std::signal(SIGINT, signalHandler);

    std::thread wait_thread([] {
        while (!gobal::exit_flag.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        c.notify_one();
    });
    {
        std::unique_lock<std::mutex> lk(mtx);
        c.wait(lk, [] { return gobal::exit_flag.load(std::memory_order_acquire); });
    }

    wait_thread.join();
    vision.stop();
    return 0;
}