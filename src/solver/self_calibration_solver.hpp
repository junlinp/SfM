#ifndef SRC_SOLVER_SELF_CALIBRATION_SOLVER_HPP_
#define SRC_SOLVER_SELF_CALIBRATION_SOLVER_HPP_
#include "eigen_alias_types.hpp"
#include "types_define.hpp"
#include "Eigen/Dense"
#include <vector>
// The Matrix K is a 3 x 3 matrix with format as follow:
//  |  alpha_x    0     cx |
//  |     0    alpha_y  cy |
//  |     0       0      1 |
//
// Such that the dual omega = KK^T
//      |   alpha_x * alpha_x + cx * cx        cx * cy      cx |
// KK^T =|     cx * cy           alpha_y * alpha_y * cy * cy cy |
//      |        cx                          cy             1  |
//
//  KK^T =   P * Q * P^T
//  We will solve Q with the constraint about the matrix KK^T
//
Eigen::Matrix4d IAC(const std::vector<Mat34>& P, size_t width, size_t height);

// row and col start with 1 instead of 0
Eigen::Matrix3d RecoveryK(const Eigen::Matrix3d& dual_omega, size_t image_width,
                          size_t image_height);

class IterativeSolver {
    private:
    double cx, cy;
    double parameters[5];
public:
    IterativeSolver(double fx = 1.0, double fy = 1.0, double cx = 0.0, double cy = 0.0, double px = 0.0, double py = 0.0, double pz = 0.0) : 
    parameters{fx, fy,px, py, pz}, cx{cx}, cy{cy} {}
    bool Solve(const std::vector<Mat34>& Ps);

    Mat33 K();

    Eigen::Vector3d p();

    Eigen::Matrix4d HomogeneousMatrix();
};

#endif  // SRC_SOLVER_SELF_CALIBRATION_SOLVER_HPP_