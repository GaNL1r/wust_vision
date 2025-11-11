cmake_minimum_required(VERSION 3.14)
project(auto_offset LANGUAGES CXX)
add_library(auto_offset OBJECT)

target_sources(auto_offset PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/auto_offset.cpp

)
