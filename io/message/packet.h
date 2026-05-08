#ifndef PACKET_H
#define PACKET_H

#include <algorithm>
#include <memory>
#include <vector>

#include "io/message/tags.h"

namespace wust_vision::message {

class Packet : public std::vector<char> {
public:
  template <typename T>
  char* change(T* ptr) {
    return reinterpret_cast<char*>(const_cast<std::add_pointer_t<std::remove_cv_t<T>>>(ptr));
  }
  template <typename T>
  bool Write(T REF_IN data) {
    insert(this->end(), change(&data), change(&data) + sizeof(data));
    return true;
  }

  template <typename T>
  bool Read(T REF_OUT data) {
    if (this->begin() + read_offset_ + sizeof(data) > this->end()) {
      return false;
    }
    std::copy_n(this->begin() + read_offset_, sizeof(data), change(&data));
    read_offset_ += sizeof(data);
    return true;
  }

  void Clear() {
    this->clear();
    read_offset_ = 0;
  }

  void Resize(const size_t n) {
    this->resize(n);
    read_offset_ = 0;
  }

  [[nodiscard]] size_t Size() const { return this->size(); }
  [[nodiscard]] bool Empty() const { return read_offset_ >= this->size(); }
  [[nodiscard]] char const* Ptr() const { return this->data(); };
  char* Ptr() { return this->data(); };

private:
  long read_offset_{};
};

};  // namespace wust_vision::message

#endif //PACKET_H