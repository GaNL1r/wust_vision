// #include "KalmanHyLib/gtsam.hpp"

// using namespace gtsam_motion_generic;

// // ------------------- ypdV2ArmorMotionModel -------------------
// struct YpdV2ArmorMotionModel {
//     double dt;

//     YpdV2ArmorMotionModel(double dt_) : dt(dt_) {}

//     gtsam::Vector operator()(const gtsam::Vector& state) const {
//         gtsam::Vector next = state;
//         assert(state.size() == 8);

//         double x = state(0), vx = state(1);
//         double y = state(2), vy = state(3);
//         double z = state(4), vz = state(5);
//         double yaw = state(6), yaw_rate = state(7);

//         next(0) = x + vx * dt;
//         next(2) = y + vy * dt;
//         next(4) = z + vz * dt;
//         next(6) = yaw + yaw_rate * dt;
//         return next;
//     }

//     void jacobian(const gtsam::Vector& state, gtsam::Matrix& J) const {
//         J.setIdentity(state.size(), state.size());
//         J(0,1) = dt; // ∂x/∂vx
//         J(2,3) = dt; // ∂y/∂vy
//         J(4,5) = dt; // ∂z/∂vz
//         J(6,7) = dt; // ∂yaw/∂yaw_rate
//     }
// };

// // ------------------- Measurement Model -------------------
// // 假设测量 yaw 和 distance = sqrt(x²+y²+z²)
// struct ArmorMeasurementModel {
//     gtsam::Vector operator()(const gtsam::Vector& state) const {
//         assert(state.size() == 8);
//         double x = state(0), y = state(2), z = state(4);
//         double yaw = state(6);
//         double dist = std::sqrt(x*x + y*y + z*z);
//         gtsam::Vector z_pred(2);
//         z_pred << yaw, dist;
//         return z_pred;
//     }

//     void jacobian(const gtsam::Vector& state, gtsam::Matrix& J) const {
//         double x = state(0), y = state(2), z = state(4);
//         double dist = std::sqrt(x*x + y*y + z*z);

//         J.setZero(2, state.size());
//         J(0,6) = 1.0;                   // ∂yaw/∂yaw
//         if (dist > 1e-6) {
//             J(1,0) = x / dist;          // ∂dist/∂x
//             J(1,2) = y / dist;          // ∂dist/∂y
//             J(1,4) = z / dist;          // ∂dist/∂z
//         }
//     }
// };
