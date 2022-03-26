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
#include "eigen_alias_types.hpp"

struct EightPointFundamentalSolverImpl {
  using DataPointType = std::pair<Observation, Observation>;
  using ModelType = Mat33;

  static constexpr size_t MINIMUM_DATA_POINT = 8;
  static constexpr size_t MODEL_FREEDOM = 2;
  static constexpr size_t MODEL_NUMBER = 1;
  static bool Fit(
      const std::vector<DataPointType>& data_points,
      ModelType* models);
};



// 
// Using Sampson Error to Ransac
// will cause DLT fails.
// chi_square_distribute[] = {
// 0.0, 3.84, 5.99, 7.82, 9.49, 11.07, 12.59, 14.07, 15.51, 16.92, 18.31};
//
struct SampsonError {
  static double Error(const std::pair<Observation, Observation>& data_point,
                      const Mat33& F) {
    Eigen::Vector3d lhs_vector, rhs_vector;
    lhs_vector = data_point.first.homogeneous(); 
    rhs_vector = data_point.second.homogeneous();

    Eigen::Vector3d rhs_f = F.transpose() * rhs_vector;
    Eigen::Vector3d lhs_f = F * lhs_vector;
    auto squared = [](double n) { return n * n; };
    double deno = squared(lhs_f(0)) + squared(lhs_f(1)) + squared(rhs_f(0)) +
                  squared(rhs_f(1));
    return squared((rhs_vector.transpose() * F).dot(lhs_vector)) / deno;
                      }

  static bool RejectRegion(double error) {
    double sigma = 1.0;
    return error * sigma * sigma > 3.84;
  }
};

struct EpipolarLineError {
  static double Error(const std::pair<Observation, Observation>& data_point, const Mat33& F) {
    Eigen::Vector3d lhs_vector, rhs_vector;
    lhs_vector = data_point.first.homogeneous(); 
    rhs_vector = data_point.second.homogeneous();

    Eigen::Vector3d rhs_f = F.transpose() * rhs_vector;
    Eigen::Vector3d lhs_f = F * lhs_vector;
    auto squared = [](double n) { return n * n; };
    double deno = (1.0 / (squared(lhs_f(0)) + squared(lhs_f(1)))) + (1.0 / (squared(rhs_f(0)) +
                  squared(rhs_f(1))));
    return squared((rhs_vector.transpose() * F).dot(lhs_vector)) * deno;
  }

  static bool RejectRegion(double error) {
    double sigma = 1.0;
    return error * sigma * sigma > 3.84;
  }
};




template<class Derived>
class FundamentalSolverInterface {

  public:
  using DataPointType = std::pair<Eigen::Vector2d, Eigen::Vector2d>;
  using ModelType = Mat33;
  bool Fit(const std::vector<DataPointType>& data_points, ModelType& model) {
    return static_cast<Derived*>(this)->Fit(data_points, model);
  }
};

// Derived Class 
// A. Ransac With eight point or seven point
// B. Gold Standand Method

class RansacEightPointFundamentalSolverInterface : public FundamentalSolverInterface<RansacEightPointFundamentalSolverInterface> {
 public:

  bool Fit(const std::vector<DataPointType>& data_points,
                  ModelType& models) {
    std::vector<size_t> placeholder;
    //return Ransac<EightPointFundamentalSolverImpl, SampsonError>::Inference(
    //    data_points, placeholder, &models);

    return Ransac<EightPointFundamentalSolverImpl, EpipolarLineError>::Inference(
        data_points, placeholder, &models);
  }

};

using EightPointFundamentalSolver = RansacEightPointFundamentalSolverInterface;

#endif  // SFM_SRC_SOLVER_FUNDAMENTAL_SOLVER_HPP_
