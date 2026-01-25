#pragma once
#include "3rdparty/backward-cpp/backward.hpp"
#include "tasks/ascii_banner.hpp"
#include "wust_vl/common/concurrency/monitored_thread.hpp"
#include "wust_vl/common/utils/signal.hpp"
#include <exception>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <utility>
template<typename T>
concept VisionLike = requires(T v) {
    {
        v.init(std::declval<bool>())
        } -> std::same_as<bool>;
    {
        v.start()
        } -> std::same_as<void>;
    {
        v.checkStateMatchMode()
        } -> std::same_as<void>;
};

template<VisionLike T>
inline int runVisionMain(int argc, char** argv) {
    printBanner();
    bool debug = false;
    if (argc > 1) {
        std::string firstArg = argv[1];
        debug = (firstArg == "true" || firstArg == "1");
        std::cout << "debug: " << firstArg << std::endl;
    }
    std::set_terminate([]() {
        std::cerr << "Uncaught exception, terminating program.\n";
        if (auto e = std::current_exception()) {
            try {
                std::rethrow_exception(e);
            } catch (const std::exception& ex) {
                std::cerr << "Exception: " << ex.what() << std::endl;
            } catch (...) {
                std::cerr << "Unknown exception" << std::endl;
            }
        }
        std::abort();
    });

    try {
        int exit_code = 0;

        {
            T v;
            v.init(debug);
            std::cout << "Starting program..." << std::endl;
            v.start();

            wust_vl::common::utils::SignalHandler sig;
            sig.start([&] {});

            bool exit_flag = false;

            while (!sig.shouldExit() && !exit_flag) {
                wust_vl::common::concurrency::ThreadManager::instance().printStatus();
                auto all_status =
                    wust_vl::common::concurrency::ThreadManager::instance().getAllThreadStatuses();
                v.checkStateMatchMode();
                for (auto& status: all_status) {
                    if (status.second == wust_vl::common::concurrency::MonitoredThread::Status::Hung) {
                        std::cerr << status.first << " is Hunging! Exiting program..." << std::endl;
                        exit_code = -1;
                        std::exit(exit_code);

                        break;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
        }

        std::cout << "Exiting program..." << std::endl;
        return exit_code;

    } catch (const std::exception& e) {
        std::cerr << "Caught exception in main: " << e.what() << "\n";
        throw;
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception caught in main!\n";
        return -1;
    }
}

#define VISION_MAIN(VISION_TYPE) \
    int main(int argc, char** argv) { \
        return runVisionMain<VISION_TYPE>(argc, argv); \
    }

#define ENABLE_BACKWARD() \
    namespace backward { \
        static backward::SignalHandling sh; \
    }