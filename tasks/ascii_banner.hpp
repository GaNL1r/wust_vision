#pragma once
#include <array>
#include <iostream>
namespace wust_vision {
constexpr std::array ascii_banner = {
    R"( _       ____  _____________   _    ___________ ________  _   __ )",
    R"(| |     / / / / / ___/_  __/  | |  / /  _/ ___//  _/ __ \/ | / /)",
    R"(| | /| / / / / /\__ \ / /     | | / // / \__ \ / // / / /  |/ / )",
    R"(| |/ |/ / /_/ /___/ // /      | |/ // / ___/ // // /_/ / /|  /  )",
    R"(|__/|__/\____//____//_/       |___/___//____/___/\____/_/ |_/   )",
};
inline void printBanner() {
    for (const auto& line: ascii_banner) {
        std::cout << line << '\n';
    }
}
} // namespace wust_vision