#ifndef INFO_H
#define INFO_H
namespace wust_vision::message {

struct GimbalSend {
  float yaw;    ///< 绝对yaw角度（度）
  float pitch;  ///< 绝对pitch角度（度）
};

struct GimbalReceive {
  float yaw;    ///< 当前绝对yaw角度（度）
  float pitch;  ///< 当前绝对pitch角度（度）
  float roll;   ///< 当前绝对roll角度（度）
  int mode;     ///< 自瞄模式 0装甲板 1小能量机关 2大能量机关
  int color;    ///< 颜色 0红色 1蓝色 2灰色 3紫色
};

struct ShootSend {
  int fire_flag;  ///< 是否开火
};

struct ShootReceive {
  float bullet_speed;  ///< 弹速
};

}  // namespace wust_vision::message
#endif //INFO_H