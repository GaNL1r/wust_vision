#ifndef VISION_SERIAL_ADAPTER_HPP
#define VISION_SERIAL_ADAPTER_HPP

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include "io/message/factory.h"
#include "io/message/info.h"
#include "io/message/message-base.h"
#include "io/message/packet.h"

enable_factory(wust_vision::message, BaseMessage, CreateMessage);

namespace wust_vision::message {

class StmMessage final : public BaseMessage {
  inline static auto registry = RegistrySub<BaseMessage, StmMessage>("stm32");

public:
  StmMessage() = default;
  ~StmMessage() override = default;
  bool Initialize(const std::string &config_path) override;
  bool Connect(bool flag) override;
  bool Send() override;
  bool Receive() override;

private:
  int serial_fd_{-1};
  bool is_physical_{true};
};

}  // namespace wust_vision::message

namespace wust_vision {

class VisionSerialAdapter {
public:
  using Ptr = std::shared_ptr<VisionSerialAdapter>;

  using ReceiveCallback = std::function<void(
      double yaw_rad, double pitch_rad, double roll_rad,
      double bullet_speed, int detect_color, int mode)>;

  VisionSerialAdapter() = default;
  ~VisionSerialAdapter();

  bool initialize(const std::string &config_path,
                  short gimbal_recv_id, short shoot_recv_id,
                  short gimbal_send_id, short shoot_send_id);

  void setReceiveCallback(ReceiveCallback cb) { receive_callback_ = std::move(cb); }

  void send(double yaw_deg, double pitch_deg, bool fire);

  void start();
  void stop();

private:
  void receiveLoop();

  std::shared_ptr<message::BaseMessage> message_;
  std::atomic<bool> running_{false};
  std::thread receive_thread_;
  ReceiveCallback receive_callback_;
};

}  // namespace wust_vision

#endif //VISION_SERIAL_ADAPTER_HPP