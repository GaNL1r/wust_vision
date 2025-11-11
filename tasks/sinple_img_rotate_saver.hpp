#pragma once
#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "wust_vl/common/utils/recorder.hpp"
#include "wust_vl/common/utils/timer.hpp"
class ImgWriter: public wust_vl::Writer<cv::Mat> {
public:
    explicit ImgWriter(
        const std::filesystem::path& video_path,
        int fps = 30,
        int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1')
    ):
        path_(video_path),
        fps_(fps),
        fourcc_(fourcc) {}

    ~ImgWriter() override {
        release();
    }

    void write(std::ostream&, const cv::Mat& frame) override {
        if (frame.empty())
            return;

        if (!initialized_) {
            initVideoWriter(frame.size());
        }

        cv::Mat bgr;
        if (frame.channels() == 3)
            cv::cvtColor(frame, bgr, cv::COLOR_RGB2BGR);
        else
            cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);

        writer_.write(bgr);
    }

private:
    void initVideoWriter(const cv::Size& frame_size) {
        if (std::filesystem::exists(path_.parent_path()) == false)
            std::filesystem::create_directories(path_.parent_path());

        writer_.open(path_.string(), fourcc_, fps_, frame_size, true);
        if (!writer_.isOpened()) {
            throw std::runtime_error("Failed to open video writer for " + path_.string());
        }
        initialized_ = true;

        std::cout << "[ImgWriter] Started writing video: " << path_ << std::endl;
    }

    void release() {
        if (initialized_) {
            writer_.release();
            std::cout << "[ImgWriter] Video saved: " << path_ << std::endl;
            initialized_ = false;
        }
    }

private:
    std::filesystem::path path_;
    cv::VideoWriter writer_;
    int fps_;
    int fourcc_;
    bool initialized_ = false;
};

class RotateWriterCSV: public wust_vl::Writer<Eigen::Vector3d> {
public:
    using RotateWriterCSVPtr = std::shared_ptr<RotateWriterCSV>;
    inline RotateWriterCSVPtr makeShared(bool write_header = true) {
        return std::make_shared<RotateWriterCSV>(write_header);
    }
    explicit RotateWriterCSV(bool write_header = true): first_write_(write_header) {}

    void write(std::ostream& os, const Eigen::Vector3d& data) override {
        double now = time_utils::sinceProgramStartSec();
        if (first_write_) {
            os << "time,yaw,pitch,roll\n";
            first_write_ = false;
        }

        os << std::fixed << std::setprecision(6) << now << "," << data[0] << "," << data[1] << ","
           << data[2] << "\n";
        os.flush();
    }

private:
    bool first_write_;
};
class RotateReaderCSV {
public:
    using RotateReaderCSVPtr = std::shared_ptr<RotateReaderCSV>;
    struct Record {
        double time;
        Eigen::Vector3d ypr;
    };

    explicit RotateReaderCSV(const std::string& csv_path, double speed = 1.0):
        csv_path_(csv_path),
        speed_(speed) {
        loadCSV();
    }

    /// 播放记录：每条数据 sleep 对齐真实时间间隔
    void replay(std::function<void(const Eigen::Vector3d&)> callback) const {
        if (records_.empty())
            return;

        for (size_t i = 0; i < records_.size(); ++i) {
            const auto& rec = records_[i];
            callback(rec.ypr);

            if (i + 1 < records_.size()) {
                double dt = (records_[i + 1].time - rec.time) / speed_;
                if (dt > 0.0)
                    std::this_thread::sleep_for(std::chrono::duration<double>(dt));
            }
        }
    }

private:
    void loadCSV() {
        std::ifstream file(csv_path_);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open CSV: " + csv_path_);
        }

        std::string line;
        bool header_skipped = false;
        while (std::getline(file, line)) {
            if (!header_skipped) { // 跳过表头
                header_skipped = true;
                continue;
            }
            std::istringstream ss(line);
            std::string token;
            Record rec;

            std::getline(ss, token, ',');
            rec.time = std::stod(token);

            for (int i = 0; i < 3; ++i) {
                std::getline(ss, token, ',');
                rec.ypr[i] = std::stod(token);
            }

            records_.push_back(rec);
        }
        std::cout << "[RotateReaderCSV] Loaded " << records_.size() << " records from " << csv_path_
                  << std::endl;
    }

private:
    std::string csv_path_;
    std::vector<Record> records_;
    double speed_ = 1.0; // 1.0=原速, 2.0=2倍速, 0.5=半速
};