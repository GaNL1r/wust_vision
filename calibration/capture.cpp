#include "Eigen/Dense"
#include "tasks/utils.hpp"
#include <fmt/core.h>

#include "tasks/packet_typedef.hpp"
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <spdlog/logger.h>
#include <wust_vl/common/drivers/serial_driver.hpp>
#include <wust_vl/video/camera.hpp>

MotionBufferGeneric<Motion, 1024> motion_buffer;
wust_vl_video::Camera camera;
void draw_text(
    cv::Mat& img,
    const std::string& text,
    const cv::Point& point,
    const cv::Scalar& color = { 0, 255, 255 },
    double font_scale = 1.0,
    int thickness = 2
) {
    cv::putText(img, text, point, cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness);
}

const std::string keys =
    "{help h usage ?  |                          | 输出命令行参数说明}"
    "{@config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
    "{output-folder o |      assets/img_with_q   | 输出文件夹路径   }";

void write_q(const std::string q_path, const Eigen::Quaterniond& q) {
    std::ofstream q_file(q_path);
    Eigen::Vector4d xyzw = q.coeffs();
    // 输出顺序为wxyz
    q_file << fmt::format("{} {} {} {}", xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
    q_file.close();
}
void write_ypr(const std::string q_path, double yaw, double pitch, double roll) {
    std::ofstream q_file(q_path);
    // 输出顺序为wxyz
    q_file << fmt::format("{} {} {}", yaw, pitch, roll);
    q_file.close();
}
void capture_loop(const std::string& config_path, const std::string& output_folder) {
    int count = 0;
    while (true) {
        auto image_frame = camera.readImage();
        image_frame.src_img = convertToMat(image_frame);
        if (image_frame.src_img.empty()) {
            continue;
        }
        auto past_att = motion_buffer.get_interpolated(image_frame.timestamp);
        double yaw = past_att->data.yaw;
        double pitch = past_att->data.pitch;
        double roll = past_att->data.roll;

        // 在图像上显示欧拉角，用来判断imuabs系的xyz正方向，同时判断imu是否存在零漂
        auto img_with_ypr = image_frame.src_img.clone();

        draw_text(img_with_ypr, fmt::format("yaw {:.2f}", yaw), { 40, 40 }, { 0, 0, 255 });
        draw_text(img_with_ypr, fmt::format("pitch{:.2f}", pitch), { 40, 80 }, { 0, 0, 255 });
        draw_text(img_with_ypr, fmt::format("rool {:.2f}", roll), { 40, 120 }, { 0, 0, 255 });

        std::vector<cv::Point2f> centers_2d;
        auto success = cv::findChessboardCorners(image_frame.src_img, cv::Size(11, 8), centers_2d);
        //auto success = cv::findCirclesGrid(image_frame.src_img, cv::Size(11, 8), centers_2d);
        cv::drawChessboardCorners(
            img_with_ypr,
            cv::Size(11, 8),
            centers_2d,
            success
        ); // 显示识别结果
        cv::resize(img_with_ypr, img_with_ypr, {}, 0.5, 0.5); // 显示时缩小图片尺寸

        // 按“s”保存图片和对应四元数，按“q”退出程序
        cv::imshow("Press s to save, q to quit", img_with_ypr);
        auto key = cv::waitKey(1);
        if (key == 'q')
            break;
        else if (key != 's')
            continue;

        // 保存图片和四元数
        count++;
        auto img_path = fmt::format("{}/{}.jpg", output_folder, count);
        auto q_path = fmt::format("{}/{}.txt", output_folder, count);
        cv::imwrite(img_path, image_frame.src_img);
        write_ypr(q_path, yaw, pitch, roll);
        std::cout << count << "Saved in" << output_folder << std::endl;
    }

    // 离开该作用域时，camera和cboard会自动关闭
}
void serialCallback(const uint8_t* data, std::size_t len) {
    try {
        std::vector<uint8_t> buf(data, data + len);
        ReceiveAimINFO aim_data = fromVector<ReceiveAimINFO>(buf);

        if (std::isnan(aim_data.roll) || std::isnan(aim_data.pitch) || std::isnan(aim_data.yaw)) {
            return;
        }

        double roll = (aim_data.roll) * M_PI / 180.0;
        double pitch = (aim_data.pitch) * M_PI / 180.0;
        double yaw = (aim_data.yaw) * M_PI / 180.0;
        double v_roll = 0.0;
        double v_pitch = 0.0;
        double v_yaw = 0.0;
        double v_x = aim_data.v_x;
        double v_y = aim_data.v_y;
        double v_z = aim_data.v_z;

        Eigen::Vector3d euler(yaw, pitch, roll);
        auto now = std::chrono::steady_clock::now();
        auto q = utils::eulerToQuat(euler, 2, 1, 0);

        Motion motion { yaw, pitch, roll, v_yaw, v_pitch, v_roll, v_x, v_y, v_z };
        motion_buffer.push(motion, now);

    } catch (const std::exception& e) {
        std::cerr << "serialCallback exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "serialCallback unknown exception" << std::endl;
    }
}
int main(int argc, char* argv[]) {
    std::string config_path = "/home/hy/wust_vision/config/camera.yaml";
    auto output_folder = "test";
    YAML::Node config = YAML::LoadFile(config_path);

    camera.init(config);
    camera.enableHikTrigger(wust_vl_video::TriggerType::Software, "Software", 1);
    camera.start();
    SerialDriver serial;
    SerialDriver::SerialPortConfig cfg { /*baud*/ 115200,
                                         /*csize*/ 8,
                                         boost::asio::serial_port_base::parity::none,
                                         boost::asio::serial_port_base::stop_bits::one,
                                         boost::asio::serial_port_base::flow_control::none };

    serial.init_port("/dev/ttyACM0", cfg);
    serial.set_receive_callback(
        std::bind(&serialCallback, std::placeholders::_1, std::placeholders::_2)
    );
    serial.set_error_callback([&](const boost::system::error_code& ec) {
        WUST_ERROR("serial") << "serial error: " << ec.message();
    });
    serial.start();
    int a = 0;
    std::cout << "wait serial init" << std::endl;
    std::cin >> a;
    // 新建输出文件夹
    std::filesystem::create_directory(output_folder);

    capture_loop(config_path, output_folder);

    return 0;
}
