#include "common/matplottools.hpp"
#include "common/logger.hpp"
#include <chrono>
#include <nlohmann/json.hpp>
#include <vector>

void write_cmd_log_to_json() {
    nlohmann::json j;
    {
        std::lock_guard<std::mutex> lock(toolsgobal::robot_cmd_mutex_);
        j["time"] = toolsgobal::time_log_;
        j["yaw"] = toolsgobal::cmd_yaw_log_;
        j["pitch"] = toolsgobal::cmd_pitch_log_;
        j["armor_dis"] = toolsgobal::armor_dis_log_;
        j["armor_x"] = toolsgobal::armor_x_log_;
        j["armor_y"] = toolsgobal::armor_y_log_;
        j["armor_z"] = toolsgobal::armor_z_log_;
        j["armor_yaw"] = toolsgobal::armor_yaw_log_;
        j["ypd_y"] = toolsgobal::ypd_y_log_;
        j["ypd_p"] = toolsgobal::ypd_p_log_;
        j["rune_obs"] = toolsgobal::rune_obs_log_;
        j["rune_pre"] = toolsgobal::rune_pre_log_;
    }

    std::ofstream file("/dev/shm/cmd_log.json");
    if (file.is_open()) {
        file << j.dump(); // 可选加上 4 缩进：j.dump(4)
    }
}
void robotCmdLoggerThread() {
    while (!gobal::is_inited_) {
        usleep(10000);
    }
    while (gobal::is_inited_) {
        write_cmd_log_to_json();
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 20Hz
    }
}
void plotRobotCmdThread() {
    while (!gobal::is_inited_) {
        usleep(10000);
    }

    matplotlibcpp::ion();
    bool figureClosed = false;

    while (gobal::is_inited_) {
        std::vector<double> time_list, yaw_list, pitch_list;
        {
            std::lock_guard<std::mutex> lock(toolsgobal::robot_cmd_mutex_);
            for (size_t i = 0; i < toolsgobal::time_log_.size(); ++i) {
                time_list.push_back(toolsgobal::time_log_[i]);
                yaw_list.push_back(toolsgobal::cmd_yaw_log_[i]);
                pitch_list.push_back(toolsgobal::cmd_pitch_log_[i]);
            }
        }

        matplotlibcpp::clf();

        matplotlibcpp::named_plot("Yaw", time_list, yaw_list, "r-");
        matplotlibcpp::named_plot("Pitch", time_list, pitch_list, "b-");

        matplotlibcpp::legend();
        matplotlibcpp::title("Robot Command Yaw and Pitch Over Time");
        matplotlibcpp::xlabel("Time (s)");
        matplotlibcpp::ylabel("Angle (rad)");
        matplotlibcpp::grid(true);
        matplotlibcpp::pause(0.001);

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    WUST_DEBUG("matplot") << "plotRobotCmdThread ending";

    // 多次尝试关闭图形窗口
    try {
        if (!figureClosed) {
            // 1. 显式关闭图形窗口
            matplotlibcpp::close();

            // 2. 确保事件循环完全停止
            // 多次处理剩余事件
            for (int i = 0; i < 10; i++) {
                matplotlibcpp::pause(0.05);
            }

            // 3. 强制关闭所有 Matplotlib 资源
            // 尝试通过 Python 命令彻底关闭
            matplotlibcpp::detail::_interpreter::get();
            PyRun_SimpleString(
                "import matplotlib.pyplot as plt\n"
                "plt.close('all')\n"
                "plt.switch_backend('agg')\n"
            ); // 切换到非交互式后端

            figureClosed = true;
            WUST_DEBUG("matplot") << "Figure closed successfully";
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception on close: " << e.what();
    }

    // 4. 确保 Python 解释器清理资源
    try {
        matplotlibcpp::detail::_interpreter::kill();
    } catch (...) {
        // 忽略可能的异常
    }

    WUST_DEBUG("matplot") << "plotRobotCmdThread fully terminated";
}

