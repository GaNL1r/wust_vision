
#include "wust_vision.hpp"
#include <csignal>
#include <mutex>
#include <string>
#include <unistd.h>

WustVision *global_vision = nullptr;
std::mutex mtx;
std::condition_variable c;
std::atomic<bool> exit_flag(false);

void signalHandler(int signum) {
  WUST_MAIN("main") << "Interrupt signal (" << signum << ") received.";
  exit_flag.store(true, std::memory_order_release);
}

int main() {
  WustVision vision;
  global_vision = &vision;
  vision.init();
  vision.run();

  std::signal(SIGINT, signalHandler);

  std::thread wait_thread([] {
    while (!exit_flag.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    c.notify_one();
  });
  {
    std::unique_lock<std::mutex> lk(mtx);
    c.wait(lk, [] { return exit_flag.load(std::memory_order_acquire); });
  }

  wait_thread.join();
  vision.stop();
  return 0;
}