#ifndef MESSAGE_BASE_H
#define MESSAGE_BASE_H

#include <unordered_map>

#include "io/message/info.h"
#include "io/message/packet.h"

namespace wust_vision::message {

class BaseMessage {
public:
  BaseMessage() = default;
  virtual ~BaseMessage() = default;

  virtual bool Initialize(const std::string &config_path) = 0;
  virtual bool Connect(bool flag) = 0;
  virtual bool Send() = 0;
  virtual bool Receive() = 0;

  template <typename T>
  void ReceiveRegister(short id) {
    Packet packet;
    packet.Resize(sizeof(T));
    packet_received_.emplace(id, std::move(packet));
    receive_size_ += sizeof(short) + sizeof(T);
  }

  template <typename T>
  void SendRegister(short id) {
    Packet packet;
    packet.Resize(sizeof(T));
    packet_sent_.emplace(id, std::move(packet));
    send_size_ += sizeof(short) + sizeof(T);
  }

  template <typename T>
  bool ReadData(T REF_OUT data) {
    short id = -1;
    for (auto &[key, packet] : packet_received_) {
      if (packet.Size() == sizeof(T)) {
        id = key;
        break;
      }
    }
    if (id == -1 || !packet_received_.contains(id)) {
      return false;
    }
    memcpy(&data, packet_received_[id].Ptr(), sizeof(T));
    return true;
  }

  template <typename T>
  bool WriteData(T REF_IN data) {
    short id = -1;
    for (auto &[key, packet] : packet_sent_) {
      if (packet.Size() == sizeof(T)) {
        id = key;
        break;
      }
    }
    if (id == -1 || !packet_sent_.contains(id)) {
      return false;
    }
    memcpy(packet_sent_[id].Ptr(), &data, sizeof(T));
    return true;
  }

protected:
  Packet send_buffer_;
  Packet receive_buffer_;
  std::unordered_map<short, Packet> packet_received_;
  std::unordered_map<short, Packet> packet_sent_;
  short receive_size_{};
  short send_size_{};
};

}  // namespace wust_vision::message

#endif //MESSAGE_BASE_H