#ifndef SRC_SOLVER_TRIANGULAR_SOLVER_HPP_
#define SRC_SOLVER_TRIANGULAR_SOLVER_HPP_

#include <vector>

#include "Eigen/Dense"

#include "eigen_alias_types.hpp"
using Mat33 = Eigen::Matrix<double, 3, 3>;

void DLT(const EigenAlignedVector<Mat34>& p_matrixs,

         const EigenAlignedVector<Eigen::Vector3d>& obs, Eigen::Vector4d& X);

void BundleAdjustmentTriangular(const EigenAlignedVector<Mat34>& p_matrixs,
                                const EigenAlignedVector<Eigen::Vector3d>& obs,
                                Eigen::Vector4d& X);

void DLTTriangular(Eigen::Vector3d& lhs_obs, Eigen::Vector3d& rhs_obs, Mat34& lhs_p, Mat34& rhs_p, Eigen::Vector4d& X);
#endif  // SRC_SOLVER_TRIANGULAR_SOLVER_HPP_