#include "3rdparty/backward-cpp/backward.hpp"
#include "tasks/vision_base.hpp"
#define COMMON_CONFIG "config/common.yaml"
#define CAMERA_CONFIG "config/camera.yaml"
#define AUTO_AIM_CONFIG "config/auto_aim.yaml"
#define AUTO_BUFF_CONFIG "config/auto_buff.yaml"

namespace backward {
backward::SignalHandling sh;
}
class vision: public VisionBase {
public:
    vision(): VisionBase(COMMON_CONFIG, CAMERA_CONFIG, AUTO_AIM_CONFIG, AUTO_BUFF_CONFIG) {}
};
VISION_MAIN(vision)