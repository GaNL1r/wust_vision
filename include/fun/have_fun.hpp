#pragma once

#include "fun/music.hpp"
#include "yaml-cpp/yaml.h"
#include <iostream>
#include <memory>
class HaveFun {
public:
    HaveFun(const YAML::Node& config) {
        music_player_ = std::make_unique<MusicPlayer>();
        std::string music_file = config["music"]["music_file"].as<std::string>();
        music_player_->loadFromFile(music_file);
        std::string timestamp_file = config["music"]["timestamp_file"].as<std::string>();
        music_player_->loadTimestampsFromCSV(timestamp_file);
        music_player_->setVolume(config["music"]["volume"].as<float>());
        if (!beep_buffer.loadFromFile("/home/hy/wust_vision/fun/bullet.wav")) {
            std::cerr << "Failed to load beep.wav" << std::endl;
        }
        beep_sound.setBuffer(beep_buffer);

        music_player_->setBeatCallback([this]() {
            // beep_sound.play(); // 播放滴声
            // std::cout << "Beat detected! 发信号啦~" << std::endl;
        });
    }
    void start() {
        music_player_->play();
    }
    void stop() {
        music_player_->stop();
    }

private:
    std::unique_ptr<MusicPlayer> music_player_;
    sf::SoundBuffer beep_buffer;
    sf::Sound beep_sound;
};