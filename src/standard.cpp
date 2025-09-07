#include "3rdparty/backward-cpp/backward.hpp"
#include "tasks/vision_base.hpp"
#define COMMON_CONFIG "config/common.yaml"
#define CAMERA_CONFIG "config/camera.yaml"
#define AUTO_AIM_CONFIG "config/auto_aim.yaml"
#define AUTO_BUFF_CONFIG "config/auto_buff.yaml"

namespace backward {
backward::SignalHandling sh;
}

class vision: public VisionBase {
public:
    vision(): VisionBase(COMMON_CONFIG, CAMERA_CONFIG, AUTO_AIM_CONFIG, AUTO_BUFF_CONFIG) {}
};
int main(int argc, char** argv) {
    std::set_terminate([]() {
        std::cerr << "Uncaught exception, terminating program.\n";
        std::abort();
    });

    try {
        vision v;
        v.init();
        v.start();

        SignalHandler sig;
        sig.start([&] { v.stop(); });

        bool exit_flag = false;
        int exit_code = 0;

        while (!sig.shouldExit() && !exit_flag) {
            wust_vl_concurrency::ThreadManager::instance().printStatus();
            auto all_status = wust_vl_concurrency::ThreadManager::instance().getAllThreadStatuses();
            v.checkStateMatchMode();

            for (auto& status: all_status) {
                if (status.second == wust_vl_concurrency::MonitoredThread::Status::Hung) {
                    std::cerr << status.first << " is Hunging! Exiting program..." << std::endl;
                    exit_flag = true;
                    exit_code = -1;
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }

        auto stop_future = std::async(std::launch::async, [&v]() { v.stop(); });
        if (stop_future.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
            std::cerr << "v.stop() timed out, forcing exit!" << std::endl;
            std::exit(1);
        }

        return exit_code;

    } catch (const std::exception& e) {
        std::cerr << "Caught exception in main: " << e.what() << "\n";
        return -1;
    } catch (...) {
        std::cerr << "Unknown exception caught in main!\n";
        return -1;
    }
}
