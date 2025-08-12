#pragma once
#include <SFML/Audio.hpp>
#include <atomic>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <sndfile.h> // libsndfile
#include <sstream>
#include <thread>
#include <vector>

class MusicPlayer: public sf::SoundStream {
public:
    using BeatCallback = std::function<void()>;

    MusicPlayer() {}

    ~MusicPlayer() {
        stop();
    }

    bool loadFromFile(const std::string& filename) {
        SF_INFO sfinfo;
        std::memset(&sfinfo, 0, sizeof(sfinfo));
        SNDFILE* sndfile = sf_open(filename.c_str(), SFM_READ, &sfinfo);
        if (!sndfile) {
            std::cerr << "Error opening audio file: " << filename << std::endl;
            return false;
        }

        samplerate = sfinfo.samplerate;
        channels = sfinfo.channels;
        totalFrames = sfinfo.frames;

        pcmDataFloat.resize(totalFrames * channels);
        sf_count_t readCount = sf_readf_float(sndfile, pcmDataFloat.data(), totalFrames);
        sf_close(sndfile);

        if (readCount != totalFrames) {
            std::cerr << "Error reading audio data: expected " << totalFrames << " frames, got "
                      << readCount << std::endl;
            return false;
        }

        convertFloatToInt16();

        initialize(channels, samplerate);

        currentFrame = 0;
        analysisPos = 0;
        currentTimestampIdx = 0;
        audioFile = filename;
        return true;
    }

    bool loadTimestampsFromCSV(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot open timestamps file: " << filename << std::endl;
            return false;
        }

        timestamps.clear();
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            float ts;
            if (ss >> ts) {
                timestamps.push_back(ts);
            }
        }
        file.close();
        currentTimestampIdx = 0;
        return true;
    }

    void setLoop(bool loop) {
        loopEnabled = loop;
    }

    void play() {
        sf::SoundStream::play();
    }

    void stop() {
        sf::SoundStream::stop();
    }

    void setBeatCallback(BeatCallback cb) {
        beatCallback = cb;
    }

    void setVolume(float volume) {
        // volume 范围是 0 ~ 100
        sf::SoundStream::setVolume(volume);
    }

    float getVolume() const {
        return sf::SoundStream::getVolume();
    }

protected:
    virtual bool onGetData(Chunk& data) override {
        std::lock_guard<std::mutex> lock(mutex);

        if (currentFrame >= totalFrames) {
            if (loopEnabled) {
                currentFrame = 0;
                analysisPos = 0;
                currentTimestampIdx = 0; // 循环时重置时间戳索引
            } else {
                return false;
            }
        }

        std::size_t samplesToStream =
            std::min(bufferSamples, (totalFrames - currentFrame)) * channels;
        data.samples = &pcmDataInt16[currentFrame * channels];
        data.sampleCount = samplesToStream;

        // 当前播放时间，单位秒
        double currentSeconds = static_cast<double>(currentFrame) / samplerate;

        // 触发所有 <= currentSeconds 的时间戳回调（支持同一帧触发多个时间戳）
        while (currentTimestampIdx < timestamps.size()
               && timestamps[currentTimestampIdx] <= currentSeconds) {
            if (beatCallback) {
                beatCallback();
            }
            ++currentTimestampIdx;
        }

        currentFrame += samplesToStream / channels;

        {
            std::lock_guard<std::mutex> analysisLock(analysisMutex);
            analysisPos = currentFrame;
        }

        return true;
    }

    virtual void onSeek(sf::Time timeOffset) override {
        std::lock_guard<std::mutex> lock(mutex);
        currentFrame = static_cast<std::size_t>(timeOffset.asSeconds() * samplerate);
        if (currentFrame >= totalFrames) {
            currentFrame = totalFrames;
        }

        // 跳转时重置时间戳索引为对应位置
        // 找到第一个大于当前时间的时间戳索引
        double currentSeconds = static_cast<double>(currentFrame) / samplerate;
        currentTimestampIdx = 0;
        while (currentTimestampIdx < timestamps.size()
               && timestamps[currentTimestampIdx] <= currentSeconds) {
            ++currentTimestampIdx;
        }

        {
            std::lock_guard<std::mutex> analysisLock(analysisMutex);
            analysisPos = currentFrame;
        }
    }

private:
    void convertFloatToInt16() {
        pcmDataInt16.resize(pcmDataFloat.size());
        for (size_t i = 0; i < pcmDataFloat.size(); i++) {
            float sample = pcmDataFloat[i];
            sample = std::clamp(sample, -1.0f, 1.0f);
            pcmDataInt16[i] = static_cast<sf::Int16>(sample * 32767.f);
        }
    }

private:
    std::vector<float> pcmDataFloat;
    std::vector<sf::Int16> pcmDataInt16;

    std::size_t currentFrame = 0;
    std::size_t totalFrames = 0;
    unsigned int samplerate = 44100;
    unsigned int channels = 2;
    bool loopEnabled = true;

    std::vector<float> timestamps;
    size_t currentTimestampIdx = 0;

    std::string audioFile;

    std::mutex mutex;
    std::mutex analysisMutex;
    std::size_t analysisPos = 0;

    BeatCallback beatCallback;

    const std::size_t bufferSamples = 4096;
};
