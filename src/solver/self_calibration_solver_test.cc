#include "gtest/gtest.h"
#include <random>
#include "Eigen/Dense"
#include "ceres/rotation.h"
#include "self_calibration_solver.hpp"

using namespace std;
TEST(SELF_CALIBRATION, IAC) {
    
    auto rand = [](double min, double max) {
        default_random_engine engine;
        uniform_real_distribution<double> uniform(min, max);
        return uniform(engine);
    };

    Eigen::Matrix3d K;
    double fx, fy, cx, cy;
    fx = rand(1.0, 1024);
    fy = rand(1.0, 1024);
    cx = rand(1024, 2048);
    cy = rand(1024, 2048);

    K << fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0;

    size_t data_size = 16;
    using Mat34 = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;
    std::vector<Mat34> data_set;
    data_set.reserve(data_size);

    for(size_t i = 0; i < data_size; i++) {
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        
        double angle_axis[3] = {rand(0.0, 180), rand(0.0, 180), rand(0.0, 180)};
        ceres::AngleAxisToRotationMatrix(angle_axis, R.data());
        t << rand(0.0, 100), rand(0.0, 100), rand(0, 100);

        Eigen::Matrix<double, 3, 4> P;
        P << R, t;
        P = K * P;
        data_set.push_back(P);
    }

    auto Q = IAC(data_set, cx * 2, cy * 2);

}
int main(int argc, char**argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}