#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/Values.h>
#include <iostream>

int main() {
    try {
        gtsam::Values vals;
        // *一定要先构造 concrete vector（不是表达式）*
        gtsam::Vector v = gtsam::Vector::Zero(11);
        vals.insert(1u, gtsam::GenericValue<gtsam::Vector>(v));
        std::cout << "insert OK\n";
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}