cmake_minimum_required(VERSION 3.14)
project(auto_sniper LANGUAGES CXX)
add_library(auto_sniper OBJECT)

target_sources(auto_sniper PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/auto_sniper.cpp

)
