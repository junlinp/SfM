#ifndef SRC_EIGEN_ALIAS_TYPES_HPP_
#define SRC_EIGEN_ALIAS_TYPES_HPP_

#include "Eigen/Dense"

using Mat34 = typename Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;

//
// More detials. see
// http://eigen.tuxfamily.org/dox/group__TopicStlContainers.html
//
template <typename T>
using EigenAlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;

#endif  // SRC_EIGEN_ALIAS_TYPES_HPP_