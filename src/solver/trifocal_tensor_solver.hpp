#ifndef SRC_SOLVER_TRIFOCAL_TENSOR_SOLVER_HPP_
#define SRC_SOLVER_TRIFOCAL_TENSOR_SOLVER_HPP_

#include <Eigen/Dense>
#include <vector>

#include "eigen_alias_types.hpp"

struct TriPair {
  Eigen::Vector3d lhs, middle, rhs;

  TriPair(const Eigen::Vector3d& lhs, const Eigen::Vector3d& middle,
          const Eigen::Vector3d& rhs)
      : lhs(lhs), middle(middle), rhs(rhs) {}
  TriPair(const Eigen::Vector2d& lhs, const Eigen::Vector2d& middle,
          const Eigen::Vector2d& rhs)
      : lhs(lhs.homogeneous()),
        middle(middle.homogeneous()),
        rhs(rhs.homogeneous()) {}
};

struct Trifocal {
  Eigen::Matrix3d lhs, middle, rhs;

  friend std::ostream& operator<<(std::ostream& os, Trifocal tirfocal);
};

std::ostream& operator<<(std::ostream& os, Trifocal tirfocal);

double Error(TriPair data_point, Trifocal model);

double GeometryError(const TriPair data_point, Trifocal& model);

void RecoveryCameraMatrix(Trifocal& trifocal, Mat34& P1, Mat34& P2, Mat34& P3);

class LinearSolver {
 public:
  using DataPointType = TriPair;
  using ModelType = Trifocal;
  void Fit(const std::vector<DataPointType>& data_points, ModelType& model);
};

class BundleRefineSolver {
 public:
  using DataPointType = TriPair;
  using ModelType = Trifocal;
  void Fit(const std::vector<DataPointType>& data_points, ModelType& model);
};
#endif  // SRC_SOLVER_TRIFOCAL_TENSOR_SOLVER_HPP_
