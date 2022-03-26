#ifndef SRC_SOLVER_EPNP_SOLVER_HPP_
#define SRC_SOLVER_EPNP_SOLVER_HPP_
#include "eigen_alias_types.hpp"
#include "pose.hpp"

namespace solver {

class EPnPSolver {
  double fx, fy, cx, cy;

 public:
  EPnPSolver(Mat33 K) : fx{K(0, 0)}, fy{K(1, 1)}, cx{K(0, 2)}, cy{K(1, 2)} {}
  EPnPSolver(double fx = 0.0, double fy = 0.0, double cx = 0.0, double cy = 0.0)
      : fx{fx}, fy{fy}, cx{cx}, cy{cy} {}
  using DataPointType = std::pair<Observation, Eigen::Vector3d>;
  using ModelType = Pose;
  bool Fit(const std::vector<DataPointType>& data_points, Pose* pose);
};
}  // namespace solver
#endif  // SRC_SOLVER_EPNP_SOLVER_HPP_