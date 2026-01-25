#include "3rdparty/backward-cpp/backward.hpp"
#include "tasks/vision_base.hpp"

ENABLE_BACKWARD()
namespace wust_vision {
class vision: public VisionBase {
public:
    vision(): VisionBase(COMMON_CONFIG, CAMERA_CONFIG, AUTO_AIM_CONFIG, AUTO_BUFF_CONFIG) {}
};
} // namespace wust_vision

VISION_MAIN(wust_vision::vision)