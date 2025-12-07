#pragma once

#include "KalmanHyLib/kalman_hybird_lib.hpp"
#include <ceres/ceres.h>

namespace imgbox_model {

static constexpr int X_N = 8; // cx vx cy vy w vw h vh
static constexpr int Z_N = 4; // measured cx cy w h

// ========================== Predict Model ==========================
struct Predict {
    Predict() = default;
    explicit Predict(double dt): dt_(dt) {}

    template<typename T>
    void operator()(const T x0[X_N], T x1[X_N]) {
        for (int i = 0; i < X_N; i++) {
            x1[i] = x0[i];
        }
        x1[0] += x0[1] * dt_; // cx
        x1[2] += x0[3] * dt_; // cy
        x1[4] += x0[5] * dt_; // w
        x1[6] += x0[7] * dt_; // h
    }

    double dt_;
};

// ========================== Measurement Model ==========================
struct Measure {
    Measure() = default;
    template<typename T>
    void operator()(const T x[X_N], T z[Z_N]) {
        z[0] = x[0]; // cx
        z[1] = x[2]; // cy
        z[2] = x[4]; // w
        z[3] = x[6]; // h
    }
};

using BBox8EKF = kalman_hybird_lib::ExtendedKalmanFilter<X_N, Z_N, Predict, Measure>;
using BBox8ESEKF = kalman_hybird_lib::ErrorStateEKF<X_N, Z_N, Predict, Measure>;

} // namespace imgbox_model
