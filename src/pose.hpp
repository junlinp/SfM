#ifndef SRC_POSE_HPP_
#define SRC_POSE_HPP_

#include "eigen_alias_types.hpp"

class Pose {
 public:
  Pose() = default;
  Pose(const Mat33& R = Mat33::Identity(),
       const Eigen::Vector3d& C = Eigen::Vector3d::Zero())
      : R_(R), C_(C) {}

 private:
  Mat33 R_;
  Eigen::Vector3d C_;
};
#endif  // SRC_POSE_HPP_