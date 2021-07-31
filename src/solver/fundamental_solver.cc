//
// Created by junlinp on 7/10/21.
//

#include "fundamental_solver.hpp"

#include <cassert>
#include <iostream>


Eigen::Matrix3d Normal(const Eigen::Matrix<double, 2, 8>& data) {

  Eigen::Matrix<double, 2, 1> center = data.rowwise().mean();
  size_t n = data.cols();
  double normal = 0.0;
  for (int i = 0; i < n; i++) {
    normal += (data.col(i) - center).norm();
  }
  //std::cout << "Normal Sum : " << (data.colwise() - center).colwise().norm().sum() << std::endl;

  double alpha = n * std::sqrt(2) / normal;
  Eigen::Matrix2d Identity = Eigen::Matrix2d::Identity();
  Eigen::Matrix3d res;
  res << alpha * Identity, alpha * center,
         0, 0, 1;
  return res;
}

void FitImpl(const Eigen::Matrix<double, 3, 8>& lhs,const Eigen::Matrix<double, 3, 8>& rhs, Eigen::Matrix3d* models) {
  Eigen::Matrix<double, 8, 9> A;

  for(int i = 0; i < 8; i++) {
    Eigen::Matrix<double, 9, 1> row;
    row << lhs(0, i) * rhs.col(i), lhs(1, i) * rhs.col(i), lhs(2, i) * rhs.col(i);
    A.row(i) = row.transpose();
  }
  Eigen::Matrix<double, 9, 9> ATA = A.transpose() * A;

  Eigen::JacobiSVD<decltype(ATA)> svd_solver(ATA, Eigen::ComputeFullV);

  //std::cout << svd_solver.singularValues()(8) << std::endl;
  
  //
  // Solve the equality : A*x = 0
  // Using SVD methods to compute Null Space
  //
  Eigen::Matrix<double, 9, 1> f_coeff = svd_solver.matrixV().col(8); 
  //std::cout << "f_coeff : " << f_coeff << std::endl;

  Eigen::Matrix3d F = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(f_coeff.data());

  Eigen::JacobiSVD<decltype(F)> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d singular = svd.singularValues();
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();
  // UDV^t
  singular(2) = 0.0;
  *models = U * singular.asDiagonal() * V.transpose();

}

void FitImpl(const Eigen::Matrix<double, 2, 8>& lhs,const Eigen::Matrix<double, 2, 8>& rhs, Eigen::Matrix3d* models) {
  Eigen::Matrix<double, 3, 8> lhs_homogous = lhs.colwise().homogeneous();
  Eigen::Matrix<double, 3, 8> rhs_homogous = rhs.colwise().homogeneous();

  Eigen::Matrix3d T = Normal(lhs);
  Eigen::Matrix3d T_dot = Normal(rhs);

  lhs_homogous = T * lhs_homogous;
  rhs_homogous = T_dot * rhs_homogous;

  FitImpl(lhs_homogous, rhs_homogous, models);

  *models = T.transpose() * (*models) * T_dot;
}

void EightPointFundamentalSolver::Fit(const std::vector<EightPointFundamentalSolver::DataPointType >& data_points, MODEL_TYPE* models) {
  assert(data_points.size() == EightPointFundamentalSolver::MINIMUM_DATA_POINT);
  Eigen::Matrix<double, 2, 8> lhs, rhs;
  for(int i = 0; i < 8; i++) {
    lhs(0, i) = data_points[i].first.x;
    lhs(1, i) = data_points[i].first.y;
    rhs(0, i) = data_points[i].second.x;
    rhs(1, i) = data_points[i].second.y;
  }
  FitImpl(lhs, rhs, models);
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
