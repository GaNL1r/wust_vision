#ifndef BUFFER_H
#define BUFFER_H

#include <mutex>

#include "io/message/tags.h"

namespace wust_vision {

template <typename T, size_t N>
class Buffer final {
public:
  Buffer() = default;
  ~Buffer() = default;

private:
  std::array<T, N> data_;
  size_t head_{};
  size_t tail_{};
  bool full_{};
  std::mutex lock_;

public:
  void Push(T FWD_IN obj) {
    std::lock_guard lock{lock_};
    data_[tail_] = std::forward<T>(obj);
    ++tail_ %= N;
    if (full_) {
      ++head_ %= N;
    }
    full_ = head_ == tail_;
  }

  bool Pop(T REF_OUT obj) {
    std::lock_guard lock{lock_};
    if (Empty()) {
      return false;
    }
    obj = std::move(data_[head_]);
    ++head_ %= N;
    full_ = false;
    return true;
  }

  [[nodiscard]] bool Empty() const { return head_ == tail_ && !full_; }

  attr_reader_val(full_, Full);
};
}  // namespace wust_vision
#endif //BUFFER_H