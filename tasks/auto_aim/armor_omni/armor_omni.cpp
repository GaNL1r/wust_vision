#include "armor_omni.hpp"
#include "3rdparty/angles.h"
#include "tasks/auto_aim/armor_tracker/motion_models/motion_modelypdv2.hpp"
#include "tasks/auto_aim/armor_tracker/target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/auto_aim/type.hpp"
#include "wust_vl/common/concurrency/ThreadPool.h"
#include "wust_vl/common/utils/timer.hpp"
#include "wust_vl/video/camera.hpp"
// clang-format off
#include "tasks/auto_aim/armor_detect/armor_detector_factory.hpp"
// clang-format on
#include "tasks/auto_aim/armor_tracker/trackerv3.hpp"
#include "tasks/auto_aim/armor_where/armor_where.hpp"
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

        void load(
            const YAML::Node& config,
            wust_vl::common::utils::Parameter::Ptr auto_aim_config_parameter
        ) noexcept {
            auto yaw_in_big_yaw_deg = config["yaw_in_big_yaw_deg"].as<double>();
            yaw_in_big_yaw = yaw_in_big_yaw_deg / 180.0 * M_PI;
            camera = std::make_shared<wust_vl::video::Camera>();
            camera->init(config);
            std::string camera_info_path =
                utils::expandEnv(config["camera_info_path"].as<std::string>());
            YAML::Node config_camera_info = YAML::LoadFile(camera_info_path);
            std::vector<double> camera_k =
                config_camera_info["camera_matrix"]["data"].as<std::vector<double>>();
            std::vector<double> camera_d =
                config_camera_info["distortion_coefficients"]["data"].as<std::vector<double>>();

            assert(camera_k.size() == 9);
            assert(camera_d.size() == 5);

            cv::Mat K(3, 3, CV_64F);
            std::memcpy(K.data, camera_k.data(), 9 * sizeof(double));

            cv::Mat D(1, 5, CV_64F);
            std::memcpy(D.data, camera_d.data(), 5 * sizeof(double));

            camera_info = std::make_pair(K.clone(), D.clone());
            auto gobal_config = auto_aim_config_parameter->getConfig();
            armor_where = ArmorWhere::create(gobal_config["armor_where"], camera_info);
            tracker = Tracker::create(auto_aim_config_parameter);
        }

        void start() noexcept {
            if (camera)
                camera->start();
        }

        int self_id;
        double total_score;
        std::shared_ptr<wust_vl::video::Camera> camera;
        ArmorWhere::Ptr armor_where;
        Tracker::Ptr tracker;
        std::pair<cv::Mat, cv::Mat> camera_info;
        double yaw_in_big_yaw;
        Target target;
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

    Impl(bool detect_color_init, const Ctx& ctx) {
        ctx_ = ctx;
        detect_color_ = detect_color_init;

        config_ = YAML::LoadFile(OMNI_CONFIG);
        auto_aim_config_parameter_ = wust_vl::common::utils::Parameter::create();
        auto_aim_config_parameter_->loadFromFile(OMNI_CONFIG);
        auto cameras = config_["cameras"].as<std::vector<std::string>>();

        for (size_t i = 0; i < cameras.size(); ++i) {
            auto real_path = utils::expandEnv(cameras[i]);
            One::Ptr one = One::create(i);
            one->load(YAML::LoadFile(real_path), auto_aim_config_parameter_);
            ones_.emplace_back(one);
        }
        auto_aim_config_parameter_->reloadFromOldPath();
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

        infer_running_count_++;

        if (common_frame.img_frame.src_img.empty()) {
            infer_running_count_--;
            return;
        }

        if (armor_detector_) {
            armor_detector_->pushInput(common_frame, std::nullopt);
        }

        infer_running_count_--;
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
        std::vector<ArmorObject> sorted_objs;

        for (const auto& obj: objs) {
            if (obj.color == ArmorColor::NONE || obj.color == ArmorColor::PURPLE) {
                continue;
            }
            sorted_objs.push_back(obj);
            std::lock_guard<std::mutex> lock(active_results_mutex_);
            active_results_.emplace_back(obj, obj.confidence, one, frame.img_frame.timestamp);
        }
        update();
        Armors armors;
        armors.timestamp = frame.img_frame.timestamp;
        Eigen::Matrix3d R_gimbal2odom = Eigen::Matrix3d::Identity();
        auto& car_b = ctx_.car_motion_buffer;
        auto& big_yaw_b = ctx_.big_yaw_motion_buffer;

        if (car_b && big_yaw_b) {
            const auto t_query = armors.timestamp;

            auto apply_motion = [&](const auto& att, const auto& att2) {
                R_gimbal2odom =
                    Eigen::AngleAxisd(
                        angles::normalize_angle(att2.data.big_yaw + one->yaw_in_big_yaw),
                        Eigen::Vector3d::UnitZ()
                    )
                    * Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY())
                    * Eigen::AngleAxisd(att.data.roll, Eigen::Vector3d::UnitX());
            };

            auto car_past_att = car_b->get_interpolated(t_query);
            auto big_yaw_past_att = big_yaw_b->get_interpolated(t_query);

            if (car_past_att && big_yaw_past_att) {
                apply_motion(*car_past_att, *big_yaw_past_att);
            } else {
                auto last_att = car_b->get_last();
                auto last_big_yaw = big_yaw_b->get_last();

                if (last_att && last_big_yaw) {
                    apply_motion(*last_att, *last_big_yaw);
                }
            }
        }
        Eigen::Matrix3d R_camera2gimbal;
        R_camera2gimbal << 0.0, 0.0, 1.0, -1.0, -0.0, 0.0, 0.0, -1.0, 0.0;
        Eigen::Matrix4d T_camera_to_odom = utils::computeCameraToOdomTransform(
            R_gimbal2odom,
            R_camera2gimbal,
            Eigen::Vector3d::Zero()
        );
        armors.armors = one->armor_where->where(sorted_objs, T_camera_to_odom);
        for (auto& armor: armors.armors) {
            armor.timestamp = armors.timestamp;
        }
        auto& target = one->target;
        target = one->tracker->track(armors);

        const auto now = std::chrono::steady_clock::now();

        const auto latency_ms =
            wust_vl::common::utils::time_utils::durationMs(frame.img_frame.timestamp, now);
        latency_averager_->add(latency_ms);
        latency_ms_ = latency_averager_->average();
        detect_count_++;

        printStats();
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
    GimbalCmd solve(double bullet_speed) {
        GimbalCmd gimbal_cmd;
        std::optional<Target> target;
        int best_target = getBestTarget();
        if (best_target < 0) {
            target = std::nullopt;
        } else {
            target = ones_[best_target]->target;
        }
        auto& very_aimer = ctx_.very_aimer;
        if (!very_aimer) {
            return gimbal_cmd;
        }
        if (target.has_value() && target->checkTargetAppear()) {
            try {
                gimbal_cmd = very_aimer->veryAim(
                    target.value(),
                    bullet_speed,
                    AutoAimFsm::AIM_WHOLE_CAR_CENTER
                );
                gimbal_cmd.enable_pitch_diff = 0.0;
                gimbal_cmd.enable_yaw_diff = 0.0;
                gimbal_cmd.fire_advice = false;
            } catch (...) {
                WUST_ERROR("omni") << "VeryAim error";
            }
        } else {
            gimbal_cmd.appear = false;
        }
        return gimbal_cmd;
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
    wust_vl::common::utils::Parameter::Ptr auto_aim_config_parameter_;

    std::unique_ptr<wust_vl::common::concurrency::Averager<double>> latency_averager_;
    int best_target_ = -1;

    int detect_count_ = 0;
    double latency_ms_;
    double min_score_ = 10.0;
    Ctx ctx_;
};

ArmorOmni::ArmorOmni(bool detect_color_init, const Ctx& ctx):
    _impl(std::make_unique<Impl>(detect_color_init, ctx)) {}

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
GimbalCmd ArmorOmni::solve(double bullet_speed) {
    return _impl->solve(bullet_speed);
}

} // namespace wust_vision::auto_aim