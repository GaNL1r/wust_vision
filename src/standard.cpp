#include "3rdparty/backward-cpp/backward.hpp"
#include "tasks/vision_base.hpp"

ENABLE_BACKWARD()
class vision: public VisionBase {
public:
    vision(): VisionBase(COMMON_CONFIG, CAMERA_CONFIG, AUTO_AIM_CONFIG, AUTO_BUFF_CONFIG) {}
};
VISION_MAIN(vision)