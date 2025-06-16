#pragma once
#include "tracker/one_tracker.hpp"
#include "tracker/one_ypd_tracker.hpp"
#include "tracker/tracker.hpp"
#include "tracker/ypd_tracker.hpp"
#include "type/type.hpp"
#include "yaml-cpp/yaml.h"
class TrackerManager {
public:
  explicit TrackerManager(const YAML::Node &config);
  void update(Target &target_, std::vector<OneTarget> &one_targets_,
              Armors armors_, std::chrono::steady_clock::time_point time);

  enum State {
    LOST,
    DETECTING,
    TRACKING,
    TEMP_LOST,
  } tracker_state;

  double dt_;
  std::unique_ptr<Tracker> tracker_;
  std::unique_ptr<OneTracker> one_tracker_;
  std::unique_ptr<YpdTracker> ypd_tracker_;
  std::unique_ptr<OneYpdTracker> one_ypd_tracker_;
  int track_one_num;
  std::vector<std::unique_ptr<OneTracker>> one_trackers_;
  double s2qx_, s2qy_, s2qz_, s2qyaw_, s2qr_, s2qd_zc_;
  double ys2qx_, ys2qy_, ys2qz_, ys2qyaw_, ys2qr_, ys2qd_zc_;
  double os2qx_, os2qy_, os2qz_, os2qyaw_;
  double oys2qx_, oys2qy_, oys2qz_, oys2qyaw_;
  double r_x_, r_y_, r_z_, r_yaw_;
  double yr_y_, yr_p_, yr_d_, yr_yaw_;
  double or_x_, or_y_, or_z_, or_yaw_;
  double oyr_y_, oyr_p_, oyr_d_, oyr_yaw_;
  double lost_time_thres_;
  double one_lost_time_thres_;
  std::chrono::steady_clock::time_point last_time_;

  std::vector<OneTarget> one_targets;
};
