#include "tasks/vision_serial_adapter.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/termios.h>
#include <unistd.h>

#include <yaml-cpp/yaml.h>

#include "wust_vl/common/utils/logger.hpp"

namespace wust_vision::message {

bool StmMessage::Initialize(const std::string &config_path) {
  auto yaml = YAML::LoadFile(config_path);
  auto control = yaml["control"];
  auto type = control["serial_type"].as<std::string>("physical");
  is_physical_ = (type == "physical");

  if (!is_physical_) {
    WUST_INFO("serial") << "serial in simulation mode";
    return true;
  }

  FILE *ls = popen("ls --color=never /dev/ttyACM*", "r");
  char name[127];
  auto ret = fscanf(ls, "%s", name);
  pclose(ls);

  if (ret == -1) {
    WUST_ERROR("serial") << "No UART device found.";
    return false;
  }

  std::string serial_port = name;
  chmod(serial_port.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);

  serial_fd_ = open(serial_port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (serial_fd_ == -1) {
    WUST_ERROR("serial") << "Failed to open serial port " << serial_port;
    return false;
  }

  termios t{};
  tcgetattr(serial_fd_, &t);
  cfmakeraw(&t);
  cfsetispeed(&t, B115200);
  tcsetattr(serial_fd_, TCSANOW, &t);

  WUST_INFO("serial") << "Serial port " << serial_port << " is open.";
  return true;
}

bool StmMessage::Connect(bool flag) {
  if (flag) {
    send_buffer_.Clear();
    send_buffer_.Write((short)0);
    for (auto &[id, _] : packet_received_) {
      send_buffer_.Write(id);
    }
  } else {
    send_buffer_.Clear();
    send_buffer_.Write((short)-1);
  }

  if (serial_fd_ < 0) {
    send_buffer_.Clear();
    return true;
  }

  std::vector<char> buffer(sizeof(short) + send_buffer_.Size());
  short size = static_cast<short>(send_buffer_.Size());
  memcpy(buffer.data(), &size, sizeof(short));
  memcpy(buffer.data() + sizeof(short), send_buffer_.Ptr(), send_buffer_.Size());

  int cnt = write(serial_fd_, buffer.data(), buffer.size());
  send_buffer_.Clear();
  return cnt == static_cast<int>(buffer.size());
}

bool StmMessage::Send() {
  if (serial_fd_ < 0) {
    send_buffer_.Clear();
    return true;
  }

  send_buffer_.Clear();
  for (auto &[id, packet] : packet_sent_) {
    send_buffer_.Write(id);
    for (long i = 0; i < packet.Size(); ++i) {
      send_buffer_.Write(packet.Ptr()[i]);
    }
  }

  std::vector<char> buffer(sizeof(short) + send_size_);
  memcpy(buffer.data(), &send_size_, sizeof(short));
  memcpy(buffer.data() + sizeof(short), send_buffer_.Ptr(), send_buffer_.Size());

  int cnt = write(serial_fd_, buffer.data(), buffer.size());
  send_buffer_.Clear();
  return cnt == static_cast<int>(buffer.size());
}

bool StmMessage::Receive() {
  if (serial_fd_ < 0) return true;

  short size = 0;
  int cnt = 0;
  auto start_time = std::chrono::steady_clock::now();

  while (cnt < static_cast<int>(sizeof(short))) {
    if (std::chrono::steady_clock::now() - start_time >= std::chrono::seconds(3)) {
      return false;
    }
    int n = read(serial_fd_, reinterpret_cast<char *>(&size) + cnt, sizeof(short) - cnt);
    if (n > 0) cnt += n;
    else if (n <= 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  if (size != receive_size_) {
    tcflush(serial_fd_, TCIFLUSH);
    return false;
  }

  std::vector<char> buf(sizeof(short) + size);
  memcpy(buf.data(), &size, sizeof(short));
  cnt = sizeof(short);

  while (cnt < static_cast<int>(buf.size())) {
    if (std::chrono::steady_clock::now() - start_time >= std::chrono::seconds(3)) {
      return false;
    }
    int n = read(serial_fd_, buf.data() + cnt, buf.size() - cnt);
    if (n > 0) cnt += n;
    else if (n <= 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  receive_buffer_.Resize(size);
  memcpy(receive_buffer_.Ptr(), buf.data() + sizeof(short), size);

  char *ptr = receive_buffer_.Ptr();
  while (ptr < receive_buffer_.Ptr() + receive_buffer_.Size()) {
    short id = *reinterpret_cast<short *>(ptr);
    ptr += sizeof(short);
    if (!packet_received_.contains(id)) break;
    auto &packet = packet_received_[id];
    memcpy(packet.Ptr(), ptr, packet.Size());
    ptr += packet.Size();
  }

  return true;
}

}  // namespace wust_vision::message

namespace wust_vision {

VisionSerialAdapter::~VisionSerialAdapter() {
  stop();
}

bool VisionSerialAdapter::initialize(const std::string &config_path,
                                      short gimbal_recv_id, short shoot_recv_id,
                                      short gimbal_send_id, short shoot_send_id) {
  message_.reset(message::CreateMessage("stm32"));
  if (!message_) {
    WUST_ERROR("serial") << "Failed to create message";
    return false;
  }

  if (!message_->Initialize(config_path)) {
    WUST_ERROR("serial") << "Failed to initialize message";
    return false;
  }

  message_->ReceiveRegister<message::GimbalReceive>(gimbal_recv_id);
  message_->ReceiveRegister<message::ShootReceive>(shoot_recv_id);
  message_->SendRegister<message::GimbalSend>(gimbal_send_id);
  message_->SendRegister<message::ShootSend>(shoot_send_id);

  if (!message_->Connect(true)) {
    WUST_ERROR("serial") << "Failed to connect";
    return false;
  }

  WUST_INFO("serial") << "VisionSerialAdapter initialized";
  return true;
}

void VisionSerialAdapter::start() {
  if (running_) return;
  running_ = true;
  receive_thread_ = std::thread(&VisionSerialAdapter::receiveLoop, this);
}

void VisionSerialAdapter::stop() {
  running_ = false;
  if (receive_thread_.joinable()) {
    receive_thread_.join();
  }
  if (message_) {
    message_->Connect(false);
  }
}

void VisionSerialAdapter::receiveLoop() {
  message::GimbalReceive gimbal_recv{};
  message::ShootReceive shoot_recv{};

  while (running_) {
    if (message_ && message_->Receive()) {
      if (message_->ReadData(gimbal_recv) && message_->ReadData(shoot_recv)) {
        if (receive_callback_) {
          double yaw_rad = gimbal_recv.yaw * M_PI / 180.0;
          double pitch_rad = gimbal_recv.pitch * M_PI / 180.0;
          double roll_rad = gimbal_recv.roll * M_PI / 180.0;

          receive_callback_(
              yaw_rad, pitch_rad, roll_rad,
              shoot_recv.bullet_speed,
              gimbal_recv.color,
              gimbal_recv.mode);
        }
      }
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

void VisionSerialAdapter::send(double yaw_deg, double pitch_deg, bool fire) {
  if (!message_) return;

  message::GimbalSend gimbal_send{static_cast<float>(yaw_deg), static_cast<float>(pitch_deg)};
  message::ShootSend shoot_send{fire ? 1 : 0};

  message_->WriteData(gimbal_send);
  message_->WriteData(shoot_send);
  message_->Send();
}

}  // namespace wust_vision