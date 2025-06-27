#include "driver/serial.hpp"
#include "common/gobal.hpp"
#include "common/logger.hpp"
#include "common/tools.hpp"
#include "driver/crc8_crc16.hpp"
#include "driver/packet_typedef.hpp"
#include "driver/sharetype.hpp"
#include "type/type.hpp"
#include <cmath>
#include <fcntl.h>
#include <iostream>
#include <pthread.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
namespace serial {

Serial::Serial()
    : device_name_(""), config_(SerialPortConfig()), is_usb_ok_(false),
      running_(false), driver_() {}
Serial::Serial(const std::string &device_name, const SerialPortConfig &config)
    : device_name_(device_name), config_(config), is_usb_ok_(false),
      running_(false), driver_() {}

Serial::~Serial() { stopThread(); }
void Serial::init(std::string device_name, SerialPortConfig config) {
  device_name_ = device_name;
  config_ = config;
}
void Serial::startThread(bool if_use_serial, bool if_use_nav) {
  this->running_ = true;
  if (if_use_nav) {
    shm_thread_ = std::thread(&Serial::shmTheard, this);
  }
  if (if_use_serial) {
    protect_thread_ = std::thread(&Serial::serialPortProtect, this);
    receive_thread_ = std::thread(&Serial::receiveData, this);
    send_thread_ = std::thread(&Serial::sendData, this);
  }
}

void Serial::stopThread() {
  this->running_ = false;
  if (protect_thread_.joinable()) {
    protect_thread_.join();
  }
  if (receive_thread_.joinable()) {
    receive_thread_.join();
  }
  if (send_thread_.joinable()) {
    send_thread_.join();
  }
  if (shm_thread_.joinable()) {
    shm_thread_.join();
  }
  if (driver_.is_open()) {
    driver_.close();
  }
}

void Serial::serialPortProtect() {
  WUST_INFO(serial_logger) << "Start serialPortProtect!";

  // 1. 初始化串口
  driver_.init_port(device_name_, config_);

  // 2. 尝试第一次打开
  try {
    if (!driver_.is_open()) {
      driver_.open();
      WUST_INFO(serial_logger) << "Serial port opened: " << device_name_;
    }
    is_usb_ok_ = true;
  } catch (const std::exception &ex) {
    WUST_ERROR(serial_logger) << "Open serial port failed: " << ex.what();
    is_usb_ok_ = false;
  }

  // 3. 循环监测，断开重连
  while (this->running_) {

    if (!is_usb_ok_) {
      try {
        if (driver_.is_open()) {
          driver_.close();
          WUST_WARN(serial_logger) << "Serial port closed for reconnect";
        }
        driver_.open();
        if (driver_.is_open()) {
          WUST_INFO(serial_logger) << "Serial port re-opened successfully";
          is_usb_ok_ = true;
        }
      } catch (const std::exception &ex) {
        is_usb_ok_ = false;
        WUST_ERROR(serial_logger)
            << "Re-open serial port failed: " << ex.what() << "\n";
      }
    }

    std::this_thread::sleep_for(
        std::chrono::milliseconds(USB_PROTECT_SLEEP_MS));
  }

  WUST_INFO(serial_logger) << "serialPortProtect stopped";
}
void Serial::receiveData() {
  WUST_INFO(serial_logger) << "receiveData started";

  std::vector<uint8_t> buffer(sizeof(ReceiveAimINFO));
  constexpr int RECEIVE_TIMEOUT_MS = 5; // 5ms 超时可调
  int retry_count = 0;

  while (this->running_) {

    if (!is_usb_ok_) {
      WUST_WARN(serial_logger)
          << "receive: USB not OK! Retry count: " << retry_count++;
      std::this_thread::sleep_for(
          std::chrono::milliseconds(USB_NOT_OK_SLEEP_MS));
      continue;
    }
    try {
      if (!driver_.receive(buffer)) {
        continue;
      }
      if (buffer.size() < sizeof(ReceiveAimINFO)) {
        WUST_WARN(serial_logger) << "receive: buffer too small, skipping";
        continue;
      }
      auto aim = fromVector<ReceiveAimINFO>(buffer);
      aim_cbk(aim);

    } catch (const std::exception &ex) {
      WUST_ERROR(serial_logger) << "receiveData exception: " << ex.what();
      is_usb_ok_ = false;
    }
  }

  WUST_INFO(serial_logger) << "receiveData stopped";
}
// void Serial::receiveData() {
//   WUST_INFO(serial_logger) << "receiveData started";

//   std::vector<uint8_t> sof_buf(1);
//   std::vector<uint8_t> header_buf;
//   std::vector<uint8_t> data_buf;

//   int sof_count = 0;
//   int retry_count = 0;

//   while (running_) {

//     if (!is_usb_ok_) {
//       WUST_WARN(serial_logger)
//           << "eceive: usb is not ok! Retry count: " << retry_count++;
//       std::this_thread::sleep_for(
//           std::chrono::milliseconds(USB_NOT_OK_SLEEP_MS));
//       continue;
//     }

//     try {

//       sof_buf.resize(39);
//       driver_.receive(sof_buf);

//       auto aim = fromVector<ReceiveAimINFO>(sof_buf);
//       aim_cbk(aim);

//     } catch (const std::exception &ex) {
//       WUST_ERROR(serial_logger) << "receiveData exception: " << ex.what();
//       is_usb_ok_ = false; // 触发重连逻辑
//     }
//   }

//   WUST_INFO(serial_logger) << "receiveData stopped";
// }

void Serial::aim_cbk(ReceiveAimINFO &aim_data) {

  static int last_reset_count = -1;

  if (std::isnan(aim_data.roll) || std::isnan(aim_data.pitch) ||
      std::isnan(aim_data.yaw) || !this->running_) {
    return;
  }

  double roll = (aim_data.roll + gobal::odom2gimbal_roll) * M_PI / 180.0;
  double pitch = (aim_data.pitch + gobal::odom2gimbal_pitch) * M_PI / 180.0;
  double yaw = (aim_data.yaw + gobal::odom2gimbal_yaw) * M_PI / 180.0;

  gobal::last_pitch = pitch;
  gobal::last_roll = roll;
  gobal::last_yaw = yaw;

  if (aim_data.manual_reset_count != last_reset_count) {
    WUST_INFO(serial_logger)
        << "Manual reset count changed: " << last_reset_count << " -> "
        << aim_data.manual_reset_count;
    gobal::if_manual_reset = true;
    last_reset_count = aim_data.manual_reset_count;
  } else {
    gobal::if_manual_reset = false;
  }

  tf::Quaternion q;
  q.setRPY(0, -pitch, yaw);

  tf::Transform gimbal_tf(tf::Position(0, 0, 0), q);
  gobal::tf_tree_.setTransform("gimbal_odom", "gimbal_link", gimbal_tf, false);

  gobal::detect_color_ = aim_data.detect_color;
  gobal::velocity = aim_data.bullet_speed;

  if (gobal::debug_mode_) {
    write_aim_log_to_json(aim_data);
  }
}

void Serial::sendData() {
  WUST_INFO(serial_logger) << "Start sendData!";

  // send_robot_cmd_data_.frame_header.sof = SOF_SEND;
  send_robot_cmd_data_.cmd_ID = ID_ROBOT_CMD;
  // send_robot_cmd_data_.frame_header.len = sizeof(SendRobotCmdData) - 6;

  //  crc8::append_CRC8_check_sum(
  //      reinterpret_cast<uint8_t *>(&send_robot_cmd_data_),
  //      sizeof(HeaderFrame));

  int retry_count = 0;

  while (this->running_) {
    if (!is_usb_ok_) {
      WUST_WARN(serial_logger)
          << "send: usb is not ok! Retry count:" << retry_count++;
      std::this_thread::sleep_for(
          std::chrono::milliseconds(USB_NOT_OK_SLEEP_MS));
      continue;
    }

    try {

      // crc16::append_CRC16_check_sum(
      //     reinterpret_cast<uint8_t *>(&send_robot_cmd_data_),
      //     sizeof(SendRobotCmdData));

      std::vector<uint8_t> send_data = toVector(send_robot_cmd_data_);
      driver_.send(send_data);
    } catch (const std::exception &ex) {
      WUST_ERROR(serial_logger) << "Error sending data: " << ex.what();
      is_usb_ok_ = false;
    }
    std::this_thread::sleep_for(
        std::chrono::milliseconds(1000 / gobal::control_rate));
  }
}
void Serial::transformGimbalCmd(GimbalCmd &gimbal_cmd, bool appear) {

  if (appear) {
    auto limit = [](double val, double max_change) {
      return std::clamp(val, -max_change, max_change);
    };

    double delta_yaw = gimbal_cmd.yaw - serial_last_yaw;
    double delta_pitch = gimbal_cmd.pitch - serial_last_pitch;

    delta_yaw = limit(delta_yaw, max_yaw_change);
    delta_pitch = limit(delta_pitch, max_pitch_change);

    send_robot_cmd_data_.yaw = serial_last_yaw + alpha_yaw * delta_yaw;
    send_robot_cmd_data_.pitch = serial_last_pitch + alpha_pitch * delta_pitch;

    serial_last_yaw = send_robot_cmd_data_.yaw;
    serial_last_pitch = send_robot_cmd_data_.pitch;
  } else {
    send_robot_cmd_data_.yaw = serial_last_yaw;
    send_robot_cmd_data_.pitch = serial_last_pitch;
  }
  // std::cout<<"yaw: "<<send_robot_cmd_data_.yaw<<" pitch:
  // "<<send_robot_cmd_data_.pitch<<std::endl;
  send_robot_cmd_data_.distance = gimbal_cmd.distance;
  send_robot_cmd_data_.pitch_diff = gimbal_cmd.pitch_diff;
  send_robot_cmd_data_.yaw_diff = gimbal_cmd.yaw_diff;
  send_robot_cmd_data_.fire = gimbal_cmd.fire_advice;
  send_robot_cmd_data_.detect_color = gobal::detect_color_;
  send_robot_cmd_data_.appear = appear;
  auto now = std::chrono::steady_clock::now();
  auto duration = now.time_since_epoch();
  uint64_t millis =
      std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
  send_robot_cmd_data_.time_stamp = static_cast<uint32_t>(millis);
}
void Serial::shmTheard() {

  while (!gobal::is_inited_) {
    usleep(10000); // 每10ms检查一次，避免占用 CPU
  }

  const char *SHM_NAME = "/cmd_vel";
  const size_t SHM_SIZE = sizeof(TwistData);

  int shm_fd = shm_open(SHM_NAME, O_RDONLY, 0666);
  if (shm_fd == -1) {
    perror("shm_open");
    WUST_ERROR(serial_logger) << "Error opening shared memory";
    return;
  }

  void *ptr = mmap(0, SHM_SIZE, PROT_READ, MAP_SHARED, shm_fd, 0);
  if (ptr == MAP_FAILED) {
    perror("mmap");
    WUST_ERROR(serial_logger) << "Error mapping shared memory";
    return;
  }

  TwistData *data = static_cast<TwistData *>(ptr);

  while (gobal::is_inited_) {

    usleep(50000); // 50ms
  }

  WUST_INFO(serial_logger) << "shmTheard end";
  return;
}
} // namespace serial