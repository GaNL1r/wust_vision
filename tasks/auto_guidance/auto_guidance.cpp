#include "auto_guidance.hpp"
#include "tasks/auto_guidance/guidance_detector/detector_base.hpp"
#include "tasks/auto_guidance/guidance_detector/detector_factory.hpp"
#include "tasks/auto_guidance/guidance_tracker/guidance_tracker.hpp"
#include "tasks/utils.hpp"
#include "wust_vl/common/concurrency/queues.hpp"
#include "wust_vl/common/utils/logger.hpp"
#include "wust_vl/common/utils/timer.hpp"

namespace auto_guidance {
struct AutoGuidance::Impl {
    ~Impl() {
        lights_queue_->stop();
        if (processing_thread_) {
            processing_thread_->stop();
            wust_vl_concurrency::ThreadManager::instance().unregisterThread(
                processing_thread_->getName()
            );
        }
    }
    void init(const YAML::Node& config, const std::pair<cv::Mat, cv::Mat>& camera_info) {
        camera_info_ = camera_info;
        std::string backend = config["backend"].as<std::string>();
        std::cout << "backend: " << backend << std::endl;
        auto detector_cfg = config["detector"];
        detector_ = DetectorFactory::createDetector(backend, detector_cfg, debug_);
        detector_->setCallback(std::bind(
            &AutoGuidance::Impl::detectCallback,
            this,
            std::placeholders::_1,
            std::placeholders::_2
        ));
        tracker_ = GuidanceTracker::create(config["tracker"]);
        lights_queue_ = std::make_unique<OrderedQueue<GreenLights>>(100, 500);
        latency_averager_ = std::make_unique<Averager<double>>(100);
    }
    void pushInput(CommonFrame& frame) {
        img_recv_count_++;
        if (detector_) {
            detector_->pushInput(frame);
        }
    }

    void detectCallback(const std::vector<GreenLight>& objs, const CommonFrame& frame) {
        detect_finish_count_++;
        GreenLights lights;
        lights.lights = objs;
        lights.timestamp = frame.timestamp;
        lights.id = frame.id;
        for (auto& light: lights.lights) {
            light.solvePnP(camera_info_.first, camera_info_.second);
            light.timestamp = frame.timestamp;
            light.image_size = frame.src_img.size();
        }
        green_lights_ = lights;
        lights_queue_->enqueue(lights);

        if (debug_) {
            dbg_.lights = lights;
            dbg_.src_img = frame.src_img;
        }
    }

    void lightsCallback(const GreenLights& lights) {
        if (lights.timestamp <= tracker_->last_time_) {
            WUST_WARN(logger_) << "Received out-of-order armor data, discarded.";
            return;
        }
        GuidanceTarget target;
        target = tracker_->track(lights);
        guidance_target_ = target;
        auto now = std::chrono::steady_clock::now();

        auto latency_ms = time_utils::durationMs(lights.timestamp, now);
        latency_averager_->add(latency_ms);
        dbg_.latency_ms = latency_averager_->average();
        if (debug_) {
            dbg_.target = target;
        }
    }
    void start() {
        processing_thread_ = wust_vl_concurrency::MonitoredThread::create(
            "AutoAimProcessingThread",
            [this](std::shared_ptr<wust_vl_concurrency::MonitoredThread> self) {
                this->processingLoop(self);
            }
        );

        wust_vl_concurrency::ThreadManager::instance().registerThread(processing_thread_);
        run_flag_ = true;
    }
    void processingLoop(std::shared_ptr<wust_vl_concurrency::MonitoredThread> self) {
        while (!self->isAlive()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        while (self->isAlive()) {
            if (!self->waitPoint())
                break;
            self->heartbeat();
            printStats();
            GreenLights lights;
            bool skip;
            if (lights_queue_->dequeue_wait(lights, skip)) {
                lightsCallback(lights);
                tracker_finish_count_++;
                if (skip) {
                    WUST_WARN(logger_) << "OrderQueue skip";
                }
            }
        }
    }
    GuidanceTarget getTarget() {
        timer_count_++;
        return guidance_target_;
    }
    void printStats() {
        utils::XSecOnce(
            [&] {
                WUST_INFO(logger_)
                    << "Rec: " << img_recv_count_ << ", Det: " << detect_finish_count_
                    << ", Fin: " << tracker_finish_count_ << ", Lat: " << dbg_.latency_ms << "ms"
                    << ", Tc:" << timer_count_;

                img_recv_count_ = 0;
                detect_finish_count_ = 0;
                tracker_finish_count_ = 0;
                timer_count_ = 0;
            },
            1.0
        );
    }
    std::unique_ptr<detector_base> detector_;
    std::string logger_ = "auto_guidance";
    std::chrono::steady_clock::time_point last_stat_time_steady_ = std::chrono::steady_clock::now();
    std::unique_ptr<GuidanceTracker> tracker_;
    bool run_flag_ = false;
    int detect_finish_count_ = 0;
    int img_recv_count_ = 0;
    int tracker_finish_count_ = 0;
    int timer_count_ = 0;
    bool debug_ = false;
    GuidanceTarget guidance_target_;
    GreenLights green_lights_;
    std::shared_ptr<wust_vl_concurrency::MonitoredThread> processing_thread_;
    std::unique_ptr<OrderedQueue<GreenLights>> lights_queue_;
    std::unique_ptr<Averager<double>> latency_averager_;
    std::pair<cv::Mat, cv::Mat> camera_info_;
    AutoGuidanceDebug dbg_;
};
AutoGuidance::AutoGuidance(): _impl(std::make_unique<Impl>()) {}
AutoGuidance::~AutoGuidance() {
    _impl.reset();
}
void AutoGuidance::init(const YAML::Node& config, const std::pair<cv::Mat, cv::Mat>& camera_info) {
    _impl->init(config, camera_info);
}
void AutoGuidance::start() {
    _impl->start();
}
void AutoGuidance::pushInput(CommonFrame& frame) {
    _impl->pushInput(frame);
}
void AutoGuidance::setDebug(bool debug) {
    _impl->debug_ = debug;
}
AutoGuidanceDebug AutoGuidance::getDebug() {
    return _impl->dbg_;
}
GuidanceTarget AutoGuidance::getTarget() {
    return _impl->getTarget();
}
} // namespace auto_guidance