// #include "geometry_msgs/msg/twist.hpp"
// #include "rclcpp/rclcpp.hpp"
// #include "ros2/ros2.hpp"
// #include <wust_vl/common/utils/signal.hpp>
// void TwistCb(const geometry_msgs::msg::Twist::SharedPtr msg) {
//     std::cout << msg->linear.x << std::endl;
// }
// int main() {
//     Ros2Node ros2;
//     ros2.add_subscription<geometry_msgs::msg::Twist>(
//         "cmd_vel",
//         std::bind(TwistCb, std::placeholders::_1),
//         rclcpp::QoS(10)
//     );
//     SignalHandler sig;
//     sig.start([&] { return 0; });
//     return 0;
// }
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>

using Photo = std::vector<std::vector<int>>;

// ------------------ 已知函数（沿用题目） ------------------
double simple_similarity(const Photo& a, const Photo& b) {
    int rows = std::min(a.size(), b.size());
    int cols = std::min(a[0].size(), b[0].size());
    double score = 0.0;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            score += 1.0 / (1 + abs(a[i][j] - b[i][j]));
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    std::cout << "match score: " << score / (rows * cols) << std::endl;
    return score / (rows * cols);
}

std::tuple<double, std::pair<int, int>> find_handsome(const Photo& img, const Photo& target) {
    double score = simple_similarity(img, target);
    return { score, { 0, 0 } };
}

Photo rot(const Photo& img, double angle) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    int rows = img.size();
    int cols = img[0].size();
    Photo out(rows, std::vector<int>(cols, 0));
    double rad = angle * M_PI / 180.0;
    double cosA = cos(rad);
    double sinA = sin(rad);
    double cx = (cols - 1) / 2.0, cy = (rows - 1) / 2.0;
    for (int y = 0; y < rows; y++)
        for (int x = 0; x < cols; x++) {
            double xt = cosA * (x - cx) + sinA * (y - cy) + cx;
            double yt = -sinA * (x - cx) + cosA * (y - cy) + cy;
            int xi = round(xt), yi = round(yt);
            if (xi >= 0 && xi < cols && yi >= 0 && yi < rows)
                out[y][x] = img[yi][xi];
        }
    std::cout << "rot done" << std::endl;
    return out;
}

class XiaoMing {
public:
    XiaoMing(const std::function<void(void)>& haoshuai, const int max_time_ms) {
        timer_thread = std::thread([&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(max_time_ms));
            std::cerr << "Warning: max_time_ms exceeded!\n";
        });
        haoshuai();
    }
    ~XiaoMing() {
        timer_thread.detach();
    }
    std::thread timer_thread;
};

// ------------------ 穷举旋转并计算总相似度 ------------------
int main() {
    std::vector<Photo> shots = {
        { { 2, 9, 9 }, { 4, 1, 9 }, { 5, 9, 9 } }, { { 9, 2, 8 }, { 5, 9, 2 }, { 6, 9, 8 } },
        { { 7, 4, 9 }, { 0, 9, 6 }, { 9, 3, 9 } }, { { 5, 2, 9 }, { 1, 9, 9 }, { 4, 9, 9 } },
        { { 6, 3, 9 }, { 0, 9, 9 }, { 3, 8, 8 } }, { { 4, 4, 9 }, { 1, 9, 9 }, { 4, 8, 9 } },
        { { 5, 9, 9 }, { 4, 1, 9 }, { 9, 2, 9 } }, { { 8, 5, 9 }, { 0, 4, 9 }, { 9, 2, 9 } },
        { { 3, 8, 9 }, { 1, 6, 9 }, { 7, 1, 7 } }, { { 4, 9, 9 }, { 1, 9, 9 }, { 2, 5, 9 } },
        { { 4, 9, 8 }, { 0, 9, 9 }, { 1, 4, 8 } }, { { 4, 7, 9 }, { 1, 7, 9 }, { 3, 5, 7 } },
        { { 3, 9, 9 }, { 0, 8, 1 }, { 5, 7, 6 } }, { { 3, 9, 9 }, { 1, 9, 5 }, { 4, 9, 4 } },
        { { 8, 3, 9 }, { 2, 8, 3 }, { 8, 5, 5 } }, { { 8, 3, 9 }, { 1, 8, 3 }, { 9, 6, 9 } }
    };

    Photo yanzhu = { { 2, 9, 9 }, { 2, 1, 4 }, { 5, 9, 9 } };

    double total_score = 0.0;
    auto start = std::chrono::high_resolution_clock::now();

    XiaoMing xiaoming(
        [&]() {
            for (const auto& photo: shots) {
                double best_score = 0.0;

                // 精细旋转 [-15°, +15°]，步长 3°
                for (double angle = -15.0; angle <= 15.0; angle += 3.0) {
                    Photo rotated = rot(photo, angle); // 精细旋转
                    auto [score, _] = find_handsome(rotated, yanzhu);
                    if (score > best_score)
                        best_score = score; // 保存最大相似度
                }

                total_score += best_score;
            }
        },
        20000
    ); // 最大 20s

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    std::cout << "Total_score: " << total_score << "  Total_time: " << elapsed << "s\n";

    return 0;
}
