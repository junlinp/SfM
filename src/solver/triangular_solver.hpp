#ifndef SRC_SOLVER_TRIANGULAR_SOLVER_HPP_
#define SRC_SOLVER_TRIANGULAR_SOLVER_HPP_

#include <vector>

#include "Eigen/Dense"
#include "eigen_alias_types.hpp"


// There is a counter example for two views triangular
// P1 = [I | 0], P2 = [0.00264714  0.00206167  0.209816   -1.875]
//                      [0.00223087   0.00173746   0.176818  -2.1876]
//                      [9.88192e-07  -1.88957e-06  0.00481149  0.000488281]

// X = [0.730447   0.682949 0.00237992  -0.00469744]
// uv1 = [306.932 286.974 1]
// uv2 = [308.084 329.19 1 ]
// P2 * X = [0.0126487  0.0135126  8.58865e-06]
double DLT(const std::vector<Mat34>& p_matrixs,
         const std::vector<Eigen::Vector3d>& obs, Eigen::Vector4d& X);

void BundleAdjustmentTriangular(const std::vector<Mat34>& p_matrixs,
                                const std::vector<Eigen::Vector3d>& obs,
                                Eigen::Vector4d& X);

void DLTTriangular(Eigen::Vector3d& lhs_obs, Eigen::Vector3d& rhs_obs,
                   Mat34& lhs_p, Mat34& rhs_p, Eigen::Vector4d& X);
#endif  // SRC_SOLVER_TRIANGULAR_SOLVER_HPP_