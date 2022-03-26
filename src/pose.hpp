#ifndef SRC_POSE_HPP_
#define SRC_POSE_HPP_

#include "eigen_alias_types.hpp"
#include "ceres/rotation.h"
class Pose {
  private: 
  double angle_axis_[3];
  double center_[3];

public:
 Pose(double* angle_axis, double* center)
     : angle_axis_{angle_axis[0], angle_axis[1], angle_axis[2]},
       center_{center[0], center[1], center[2]} {}

 Pose(Mat33 R = Mat33::Identity(), Eigen::Vector3d C = Eigen::Vector3d::Zero()) : angle_axis_{0}, center_{C.x(), C.y(), C.z()} {
      ceres::RotationMatrixToAngleAxis(R.data(), angle_axis_);
 }

 double* angle_axis() { return &angle_axis_[0]; }

 double* center() { return &center_[0]; }

 Mat33 R() {
      Mat33 r;
      ceres::AngleAxisToRotationMatrix(angle_axis_, r.data());
      return r;
 }

 Eigen::Vector3d C() {
      return Eigen::Vector3d(center_[0], center_[1], center_[2]);
 }

 Mat34 P() {
      Mat34 p;
      p << R(), -R() * C();
      return p;
 }

};
#endif  // SRC_POSE_HPP_