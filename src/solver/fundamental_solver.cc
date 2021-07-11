//
// Created by junlinp on 7/10/21.
//

#include "fundamental_solver.hpp"

#include <cassert>
#include <iostream>

void EightPointFundamentalSolver::Fit(const std::vector<EightPointFundamentalSolver::DataPointType >& data_points, MODEL_TYPE* models) {
  assert(data_points.size() == EightPointFundamentalSolver::MINIMUM_DATA_NUMBER);
  Eigen::Matrix<double, 8, 9> A;
  for(int i = 0; i < 8; i++) {
    double lhs_x = data_points[i].first.x;
    double lhs_y = data_points[i].first.y;
    double rhs_x = data_points[i].second.x;
    double rhs_y = data_points[i].second.y;
    A(i, 0) = lhs_x * rhs_x;
    A(i, 1) = lhs_x * rhs_y;
    A(i, 2) = lhs_x * 1.0;


    A(i, 3) = lhs_y * rhs_x;
    A(i, 4) = lhs_y * rhs_y;
    A(i, 5) = lhs_y * 1.0;


    A(i, 6) = 1.0 * rhs_x;
    A(i, 7) = 1.0 * rhs_y;
    A(i, 8) = 1.0 * 1.0;
  }

  Eigen::Matrix<double, 9, 9> ATA = A.transpose() * A;

  Eigen::JacobiSVD<decltype(ATA)> svd_solver(ATA, Eigen::ComputeFullV);

  std::cout << svd_solver.singularValues()(8) << std::endl;
  
  //
  // Solve the equality : A*x = 0
  // Using SVD methods to compute Null Space
  //
  Eigen::Matrix<double, 9, 1> f_coeff = svd_solver.matrixV().col(8); 
  std::cout << "f_coeff : " << f_coeff << std::endl;

  Eigen::Matrix3d F = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(f_coeff.data());

  Eigen::JacobiSVD<decltype(F)> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d singular = svd.singularValues();
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();
  // UDV^t
  singular(2) = 0.0;
  *models = U * singular.asDiagonal() * V.transpose();
}

double EightPointFundamentalSolver::Error(const DataPointType& data_point, const MODEL_TYPE& model) {
  // sampson error

  KeyPoint lhs = data_point.first;
  KeyPoint rhs = data_point.second;
  Eigen::Vector3d lhs_vector, rhs_vector;
  lhs_vector << lhs.x, lhs.y, 1.0;
  rhs_vector << rhs.x, rhs.y, 1.0;

  Eigen::Vector3d lhs_f = lhs_vector.transpose() * model;
  Eigen::Vector3d rhs_f = rhs_vector.transpose() * model.transpose();
  auto squared = [](double n) { return n * n;};
  double deno = squared(lhs_f(0)) + squared(lhs_f(1)) + squared(rhs_f(0)) + squared(rhs_f(1));
  return squared((lhs_vector.transpose() * model).dot(rhs_vector)) / deno;
}
