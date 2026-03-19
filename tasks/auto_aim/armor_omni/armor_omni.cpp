#include "armor_omni.hpp"
#include "tasks/auto_aim/type.hpp"
#include "wust_vl/common/concurrency/ThreadPool.h"
#include "wust_vl/common/utils/timer.hpp"
#include "wust_vl/video/camera.hpp"
// clang-format off
#include "tasks/auto_aim/armor_detect/armor_detector_factory.hpp"
#include <opencv2/highgui.hpp>
// clang-format on

namespace wust_vision::auto_aim {

struct ArmorOmni::Impl {
    struct One {
        using Ptr = std::shared_ptr<One>;

        One(int id) {
            self_id = id;
            total_score = 0;
        }

        static Ptr create(int id) {
            return std::make_shared<One>(id);
        }

        void load(const YAML::Node& config) noexcept {
            camera = std::make_shared<wust_vl::video::Camera>();
            camera->init(config);
        }

        void start() noexcept {
            if (camera)
                camera->start();
        }

        int self_id;
        double total_score;
        std::shared_ptr<wust_vl::video::Camera> camera;
    };

    struct Obj {
        ArmorObject armor;
        double score = 0;
        One::Ptr one;
        std::chrono::steady_clock::time_point timestamp;

        Obj(const ArmorObject& a,
            double s,
            const One::Ptr& o,
            std::chrono::steady_clock::time_point ts):
            armor(a),
            score(s),
            one(o),
            timestamp(ts) {
            if (one)
                one->total_score += score;
        }

        ~Obj() {
            if (one)
                one->total_score -= score;
        }
    };

    static constexpr const char* _ML_CONFIG = "config/omni/detect_ml.yaml";
    static constexpr const char* _OPENCV_CONFIG = "config/omni/detect_opencv.yaml";

    ~Impl() {
        run_flag_ = false;
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
        min_score_ = config_["min_score"].as<double>();
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

        timer_ = std::make_unique<wust_vl::common::utils::Timer>("omni");
        latency_averager_ = std::make_unique<wust_vl::common::concurrency::Averager<double>>(100);
    }

    void start() noexcept {
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

    void timerCallback(double dt_ms) noexcept {
        if (!run_flag_ || main_tracking_)
            return;

        int one_id = getOneId();

        auto& one = ones_[one_id];

        auto frame = one->camera->readImage();

        if (frame.src_img.empty())
            return;

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
        if (infer_running_count_ >= max_infer_running_ || !thread_pool_ || !run_flag_) {
            return;
        }

        thread_pool_->enqueue([this, frame = std::move(common_frame)]() mutable {
            if (!run_flag_ || ones_.empty())
                return;

            infer_running_count_++;

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

    void setDetectColor(bool flag) noexcept {
        detect_color_ = flag;
    }

    void updateMainTracking(bool flag) noexcept {
        main_tracking_ = flag;
    }

    int getBestTarget() noexcept {
        update();
        return best_target_;
    }

    void
    ArmorDetectCallback(const std::vector<ArmorObject>& objs, const CommonFrame& frame) noexcept {
        auto one = std::any_cast<One::Ptr>(frame.any_ctx);

        for (const auto& obj: objs) {
            if (obj.color == ArmorColor::NONE || obj.color == ArmorColor::PURPLE) {
                continue;
            }
            std::lock_guard<std::mutex> lock(active_results_mutex_);
            active_results_.emplace_back(obj, obj.confidence, one, frame.img_frame.timestamp);
        }
        // if(one->self_id==1)
        // {
        //     cv::imshow("a",frame.img_frame.src_img);
        //     cv::waitKey(1);
        // }
        const auto now = std::chrono::steady_clock::now();

        const auto latency_ms =
            wust_vl::common::utils::time_utils::durationMs(frame.img_frame.timestamp, now);
        latency_averager_->add(latency_ms);
        latency_ms_ = latency_averager_->average();
        detect_count_++;

        printStats();

        update();
    }

    void update() noexcept {
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

        if (active_results_.empty()) {
            best_target_ = -1;
            return;
        }

        double max_score = min_score_;
        best_target_ = -1;
        for (size_t i = 0; i < ones_.size(); ++i) {
            if (ones_[i]->total_score > max_score) {
                max_score = ones_[i]->total_score;
                best_target_ = ones_[i]->self_id;
            }
        }
    }

    void printStats() {
        utils::XSecOnce(
            [&] {
                WUST_INFO("armor_omni") << "det: " << detect_count_ << " best: " << best_target_
                                        << " lat: " << latency_ms_;

                detect_count_ = 0;
            },
            1.0
        );
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
    std::unique_ptr<wust_vl::common::concurrency::Averager<double>> latency_averager_;
    int best_target_ = -1;

    int detect_count_ = 0;
    double latency_ms_;
    double min_score_ = 10.0;
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