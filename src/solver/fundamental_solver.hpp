//
// Created by junlinp on 7/10/21.
//

#ifndef SFM_SRC_SOLVER_FUNDAMENTAL_SOLVER_HPP_
#define SFM_SRC_SOLVER_FUNDAMENTAL_SOLVER_HPP_

#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "keypoint.hpp"

#include "ransac.hpp"

namespace {
  using Observation = Eigen::Vector2d;

}

struct EightPointFundamentalSolverImpl {
  using DataPointType = std::pair<Observation, Observation>;
  using ModelType = Eigen::Matrix3d;

  static constexpr size_t MINIMUM_DATA_POINT = 8;
  static constexpr size_t MODEL_FREEDOM = 4;
  static constexpr size_t MODEL_NUMBER = 1;
  static void Fit(
      const std::vector<std::pair<Observation, Observation>, Eigen::aligned_allocator<DataPointType>>& data_points,
      Eigen::Matrix3d* models);
};

struct SampsonError {
  static double Error(const std::pair<Observation, Observation>& data_point,
                      const Eigen::Matrix3d& F) {
    Eigen::Vector3d lhs_vector, rhs_vector;
    lhs_vector = data_point.first.homogeneous(); 
    rhs_vector = data_point.second.homogeneous();

    Eigen::Vector3d rhs_f = rhs_vector.transpose() * F;
    Eigen::Vector3d lhs_f = F.transpose() * lhs_vector;
    auto squared = [](double n) { return n * n; };
    double deno = squared(lhs_f(0)) + squared(lhs_f(1)) + squared(rhs_f(0)) +
                  squared(rhs_f(1));
    return squared((rhs_vector.transpose() * F).dot(lhs_vector)) / deno;
                      }
};




template<class Derived>
class FundamentalSolverInterface {

  public:
  using DataPointType = std::pair<Eigen::Vector2d, Eigen::Vector2d>;
  using ModelType = Eigen::Matrix3d;
  bool Fit(const std::vector<DataPointType>& data_points, ModelType& model) {
    return static_cast<Derived*>(this)->Fit(data_points, model);
  }
};

// Derived Class 
// A. Ransac With eight point or seven point
// B. Gold Standand Method

class RansacEightPointFundamentalSolverInterface : public FundamentalSolverInterface<RansacEightPointFundamentalSolverInterface> {
 public:

  bool Fit(const std::vector<DataPointType, Eigen::aligned_allocator<DataPointType>>& data_points,
                  ModelType& models) {
    std::vector<size_t> placeholder;
    return Ransac<EightPointFundamentalSolverImpl, SampsonError>::Inference(
        data_points, placeholder, &models);
  }
};

using EightPointFundamentalSolver = RansacEightPointFundamentalSolverInterface;

#endif  // SFM_SRC_SOLVER_FUNDAMENTAL_SOLVER_HPP_
