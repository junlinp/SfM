//
// Created by junlinp on 7/10/21.
//

#ifndef SFM_SRC_SOLVER_FUNDAMENTAL_SOLVER_HPP_
#define SFM_SRC_SOLVER_FUNDAMENTAL_SOLVER_HPP_

#include "Eigen/Dense"
#include "keypoint.hpp"
#include <vector>
#include <utility>

class EightPointFundamentalSolver {
 public:
  using DataPointType = std::pair<KeyPoint, KeyPoint>;
  static constexpr size_t MINIMUM_DATA_NUMBER = 8;
  static constexpr size_t MODEL_FREEDOM = 1;
  static constexpr size_t MODEL_NUMBER = 1;
  using MODEL_TYPE = Eigen::Matrix3d;

  static void Fit(const std::vector<DataPointType>& data_points, MODEL_TYPE* models);
  static double Error(const DataPointType& data_point, const MODEL_TYPE& model);
};

#endif //SFM_SRC_SOLVER_FUNDAMENTAL_SOLVER_HPP_
