#ifndef SRC_SOLVER_TRIANGULAR_SOLVER_HPP_
#define SRC_SOLVER_TRIANGULAR_SOLVER_HPP_

#include <vector>

#include "Eigen/Dense"

using Mat34 = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;
template<typename T>
using EigenAlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;

void DLT(const EigenAlignedVector<Mat34>& p_matrixs,

         const EigenAlignedVector<Eigen::Vector3d>& obs, Eigen::Vector4d& X);

void BundleAdjustmentTriangular(const EigenAlignedVector<Mat34>& p_matrixs,
                                const EigenAlignedVector<Eigen::Vector3d>& obs,
                                Eigen::Vector4d& X);

#endif  // SRC_SOLVER_TRIANGULAR_SOLVER_HPP_