// pch.hpp
#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <thread>
#include <exception>
#include <utility>
#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>
#include <array>
#include <numeric>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fmt/core.h>
#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>


#include "3rdparty/angles.h"
#include "3rdparty/backward-cpp/backward.hpp"