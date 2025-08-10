
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

#include "wust_vision.hpp"
#include <atomic>
#include <condition_variable>
#include <csignal>
#include <iostream>
#include <mutex>
#include <thread>

// 全局退出标志
std::atomic<bool> exit_flag(false);
std::atomic<bool> sigint_received(false);

std::mutex mtx;
std::condition_variable c;

void signalHandler(int signum) {
    if (!sigint_received.exchange(true)) {
        WUST_MAIN("main") << "Interrupt signal (" << signum << ") received. Exiting gracefully...";
        exit_flag.store(true, std::memory_order_release);
    } else {
        WUST_MAIN("main") << "Interrupt signal (" << signum << ") received again. Forcing exit.";
        std::exit(EXIT_FAILURE); // 立刻退出，不执行清理
    }
}

int main() {
    const char* ld_path = std::getenv("LD_LIBRARY_PATH");
    std::string ld_path_str = ld_path ? ld_path : "";

    WUST_MAIN("main"
    ) << "环境变量：\n"
      << "MVCAM_SDK_PATH="
      << (std::getenv("MVCAM_SDK_PATH") ? std::getenv("MVCAM_SDK_PATH") : "NULL") << '\n'
      << "MVCAM_COMMON_RUNENV="
      << (std::getenv("MVCAM_COMMON_RUNENV") ? std::getenv("MVCAM_COMMON_RUNENV") : "NULL") << '\n'
      << "MVCAM_GENICAM_CLPROTOCOL="
      << (std::getenv("MVCAM_GENICAM_CLPROTOCOL") ? std::getenv("MVCAM_GENICAM_CLPROTOCOL") : "NULL"
         )
      << '\n'
      << "ALLUSERSPROFILE="
      << (std::getenv("ALLUSERSPROFILE") ? std::getenv("ALLUSERSPROFILE") : "NULL") << '\n'
      << "LD_LIBRARY_PATH=" << ld_path_str;

    const std::string required_path = "/opt/MVS/lib/64:/opt/MVS/lib/32";
    if (ld_path_str.find(required_path) == std::string::npos) {
        WUST_ERROR("main") << "错误: LD_LIBRARY_PATH 不包含必需路径: " << required_path;
        system(" export LD_LIBRARY_PATH=/opt/MVS/lib/64:/opt/MVS/lib/32:$LD_LIBRARY_PATH ");
        WUST_INFO("main"
        ) << "运行： export LD_LIBRARY_PATH=/opt/MVS/lib/64:/opt/MVS/lib/32:$LD_LIBRARY_PATH";
        return EXIT_FAILURE;
    }

    WustVision vision;
    if (vision.init()) {
        vision.start();
    }

    std::signal(SIGINT, signalHandler);

    std::thread wait_thread([] {
        while (!exit_flag.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        c.notify_one();
    });

    {
        std::unique_lock<std::mutex> lk(mtx);
        c.wait(lk, [] { return exit_flag.load(std::memory_order_acquire); });
    }

    wait_thread.join();
    vision.stop();

    return 0;
}
