#include "armor_omni.hpp"
#include "wust_vl/common/concurrency/ThreadPool.h"
#include "wust_vl/common/utils/timer.hpp"
#include "wust_vl/video/camera.hpp"
// clang-format off
#include "tasks/auto_aim/armor_detect/armor_detector_factory.hpp"
// clang-format on
namespace wust_vision::auto_aim {
struct ArmorOmni::Impl {
    struct One {
        using Ptr = std::shared_ptr<One>;
        One(int id) {
            self_id = id;
        }
        static Ptr create(int id) {
            return std::make_shared<One>(id);
        }
        void load(const YAML::Node& config) {
            camera = std::make_shared<wust_vl::video::Camera>();
            camera->init(config);
        }
        void start() {
            camera->start();
        }
        int self_id;
        double total_score = 0;
        std::shared_ptr<wust_vl::video::Camera> camera;
    };
    static constexpr const char* OMNI_CONFIG = "config/omni/omni.yaml";
    static constexpr const char* _ML_CONFIG = "config/omni/detect_ml.yaml";
    static constexpr const char* _OPENCV_CONFIG = "config/omni/detect_opencv.yaml";
    ~Impl() {
        run_flag_ = false;
        thread_pool_->waitUntilEmpty();
        if (armor_detector_) {
            armor_detector_.reset();
        }
    }

    Impl(bool detect_color_init) {
        detect_color_ = detect_color_init;
        config_ = YAML::LoadFile(OMNI_CONFIG);
        auto cameras = config_["cameras"].as<std::vector<std::string>>();
        for (size_t i = 0; i < cameras.size(); ++i) {
            One::Ptr one = One::create(i);
            one->load(YAML::LoadFile(cameras[i]));
            ones_.emplace_back(one);
        }
        fps_ = config_["fps"].as<int>(30);
        active_time_ = config_["active_time"].as<double>(0.5);
        max_infer_running_ = config_["max_infer_running"].as<int>(0);
        const std::string armor_detect_backend =
            config_["armor_detect_backend"].as<std::string>("");
        armor_detector_ = DetectorFactory::createArmorDetector(
            armor_detect_backend,
            false,
            _OPENCV_CONFIG,
            _ML_CONFIG
        );
        armor_detector_->setCallback(std::bind(
            &ArmorOmni::Impl::ArmorDetectCallback,
            this,
            std::placeholders::_1,
            std::placeholders::_2
        ));
        thread_pool_ =
            std::make_unique<wust_vl::common::concurrency::ThreadPool>(max_infer_running_);
        timer_ = std::make_unique<wust_vl::common::utils::Timer>();
    }
    struct Obj {
        ArmorObject armor;
        One::Ptr one;
        std::chrono::steady_clock::time_point timestamp;
    };
    void start() {
        run_flag_ = true;
        for (auto& one: ones_) {
            one->start();
        }
        if (timer_) {
            const auto timercallback =
                std::bind(&ArmorOmni::Impl::timerCallback, this, std::placeholders::_1);
            const double rate_hz = fps_;
            timer_->start(rate_hz, timercallback);
        }
    }
    int getOneId() const {
        static int one_id = 0;
        int id = one_id;
        one_id = (one_id + 1) % ones_.size();
        return id;
    }
    void timerCallback(double dt_ms) {
        if (!run_flag_ || main_tracking_) {
            return;
        }
        int one_id = getOneId();
        auto& one = ones_[one_id];
        auto frame = one->camera->readImage();
        if (frame.src_img.empty()) {
            return;
        }
        CommonFrame common_frame;
        common_frame.img_frame = frame;
        common_frame.id = one_id;
        common_frame.detect_color = detect_color_;
        common_frame.expanded = cv::Rect(0, 0, frame.src_img.cols, frame.src_img.rows);
        common_frame.offset = cv::Point2f(0, 0);
        common_frame.any_ctx = one;
        detect(common_frame);
    }
    void detect(CommonFrame& common_frame) {
        thread_pool_->enqueue([this, frame = std::move(common_frame)]() mutable {
            infer_running_count_++;
            if (frame.img_frame.src_img.data == nullptr) {
                infer_running_count_--;
                return;
            }
            if (frame.img_frame.src_img.empty()) {
                infer_running_count_--;
                return;
            }
            if (armor_detector_) {
                armor_detector_->pushInput(frame, std::nullopt);
            }
            infer_running_count_--;
        });
    }

    void setDetectColor(bool flag) {
        detect_color_ = flag;
    }
    void updateMainTracking(bool flag) {
        main_tracking_ = flag;
    }
    int getBestTarget() {
        update();
        int best_target = -1;
        double max_score = -1;
        if (active_results_.empty()) {
            return best_target;
        }
        for (int i = 0; i < ones_.size(); ++i) {
            if (ones_[i]->total_score > max_score) {
                max_score = ones_[i]->total_score;
                best_target = i;
            }
        }
        return best_target;
    }
    void ArmorDetectCallback(const std::vector<ArmorObject>& objs, const CommonFrame& frame) {
        auto one = std::any_cast<One::Ptr>(frame.any_ctx);
        std::cout << "one_id: " << one->self_id << std::endl;
        for (auto& obj: objs) {
            Obj _obj;
            _obj.armor = obj;
            _obj.one = one;
            _obj.timestamp = frame.img_frame.timestamp;
            std::lock_guard<std::mutex> lock(active_results_mutex_);
            active_results_.push_back(std::move(_obj));
        }
        update();
    }
    void update() {
        std::lock_guard<std::mutex> lock(active_results_mutex_);
        while (!active_results_.empty()) {
            auto& obj = active_results_.front();
            if (std::abs(wust_vl::common::utils::time_utils::durationSec(
                    obj.timestamp,
                    wust_vl::common::utils::time_utils::now()
                ))
                > active_time_)
            {
                active_results_.pop_front();
            } else {
                break;
            }
        }
        for (auto& one: ones_) {
            one->total_score = 0;
        }
        for (auto& obj: active_results_) {
            obj.one->total_score += obj.armor.confidence;
        }
    }
    int fps_;
    int max_infer_running_ = 0;
    std::atomic<int> infer_running_count_ { 0 };
    bool detect_color_;
    bool main_tracking_ = false;
    bool run_flag_ = false;
    double active_time_ = 0;
    std::deque<Obj> active_results_;
    mutable std::mutex active_results_mutex_;
    std::vector<One::Ptr> ones_;
    YAML::Node config_;
    std::unique_ptr<wust_vl::common::concurrency::ThreadPool> thread_pool_;
    std::unique_ptr<wust_vl::common::utils::Timer> timer_;
    ArmorDetectorBase::Ptr armor_detector_;
};
ArmorOmni::ArmorOmni(bool detect_color_init): _impl(std::make_unique<Impl>(detect_color_init)) {}
ArmorOmni::~ArmorOmni() {
    _impl.reset();
}
void ArmorOmni::start() {
    _impl->start();
}
void ArmorOmni::setDetectColor(bool flag) {
    _impl->setDetectColor(flag);
}
void ArmorOmni::updateMainTracking(bool flag) {
    _impl->updateMainTracking(flag);
}
int ArmorOmni::getBestTarget() {
    return _impl->getBestTarget();
}
} // namespace wust_vision::auto_aim