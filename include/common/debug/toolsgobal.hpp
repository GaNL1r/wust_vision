#pragma once

#include "common/tf.hpp"
#include "detect/mono_measure_tool.hpp"
#include "tracker/tracker.hpp"
#include <thread>
class LatencyAveragerDeque {
public:
    explicit LatencyAveragerDeque(size_t window_size = 100): window_size_(window_size) {}

    void add(int64_t latency_ns) {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_.push_back(latency_ns);
        if (buffer_.size() > window_size_) {
            buffer_.pop_front();
        }
    }

    double average_ms() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffer_.empty())
            return 0.0;
        int64_t sum = 0;
        for (auto val: buffer_)
            sum += val;
        return static_cast<double>(sum) / buffer_.size() / 1e6;
    }

    void setWindowSize(size_t new_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        window_size_ = new_size;
        while (buffer_.size() > window_size_) {
            buffer_.pop_front();
        }
    }

private:
    mutable std::mutex mutex_;
    std::deque<int64_t> buffer_;
    size_t window_size_;
};
struct DebugLogs {
    std::vector<double> time_log;
    std::vector<double> cmd_yaw_log;
    std::vector<double> cmd_pitch_log;
    std::vector<double> armor_dis_log;
    std::vector<double> armor_x_log;
    std::vector<double> armor_y_log;
    std::vector<double> armor_z_log;
    std::vector<double> armor_yaw_log;
    std::vector<double> ypd_y_log;
    std::vector<double> ypd_p_log;
    std::vector<double> rune_obs_log;
    std::vector<double> rune_pre_log;
    std::vector<double> rune_v_log;

    void clear() {
        time_log.clear();
        cmd_yaw_log.clear();
        cmd_pitch_log.clear();
        armor_dis_log.clear();
        armor_x_log.clear();
        armor_y_log.clear();
        armor_z_log.clear();
        armor_yaw_log.clear();
        ypd_y_log.clear();
        ypd_p_log.clear();
        rune_obs_log.clear();
        rune_pre_log.clear();
        rune_v_log.clear();
    }

    void push_back(
        double time,
        double cmd_yaw,
        double cmd_pitch,
        double armor_dis,
        double armor_x,
        double armor_y,
        double armor_z,
        double armor_yaw,
        double ypd_y,
        double ypd_p,
        double rune_obs,
        double rune_pre,
        double rune_v
    ) {
        time_log.push_back(time);
        cmd_yaw_log.push_back(cmd_yaw);
        cmd_pitch_log.push_back(cmd_pitch);
        armor_dis_log.push_back(armor_dis);
        armor_x_log.push_back(armor_x);
        armor_y_log.push_back(armor_y);
        armor_z_log.push_back(armor_z);
        armor_yaw_log.push_back(armor_yaw);
        ypd_y_log.push_back(ypd_y);
        ypd_p_log.push_back(ypd_p);
        rune_obs_log.push_back(rune_obs);
        rune_pre_log.push_back(rune_pre);
        rune_v_log.push_back(rune_v);
    }
};

struct DebugArmor {
    std::optional<imgframe> src_img;
    std::optional<armor::Armors> armors;
    std::optional<Target_info> target_info;
    std::optional<armor::Target> target;
    std::optional<Tracker::State> tracker_state;
    std::optional<GimbalCmd> gimbal_cmd;
};
struct DebugRune {
    std::optional<imgframe> src_img;
    std::optional<std::vector<rune::RuneObject>> objs;
    std::optional<double> predict_angle;
    std::optional<GimbalCmd> gimbal_cmd;
    std::optional<std::string> debug_text;
    std::optional<std::vector<cv::Point2f>> manual_r_box;
};

namespace toolsgobal {

extern std::mutex robot_cmd_mutex_;
extern DebugLogs debug_logs_;
extern int debug_w;
extern int debug_h;
extern double debug_fps;
extern double latency_ms;
} // namespace toolsgobal
