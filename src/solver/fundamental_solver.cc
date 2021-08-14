//
// Created by junlinp on 7/10/21.
//

#include "fundamental_solver.hpp"

#include <cassert>
#include <iostream>
// TODO (junlinp) :
// Error will not be zero for the samples used to Fit the model.

Eigen::Matrix3d Normal(const Eigen::Matrix<double, 2, 8>& data) {

  Eigen::Matrix<double, 2, 1> center = data.rowwise().mean();
  size_t n = data.cols();
  double normal = (data.colwise() - center).colwise().norm().sum();

  double alpha = n * std::sqrt(2.0) / normal;
  Eigen::Matrix3d res;
  res << alpha ,     0.0, -alpha * center(0),
         0.0   ,   alpha, -alpha * center(1),
         0.0   ,     0.0, 1.0;
  return res;
}

void FitImpl(const Eigen::Matrix<double, 3, 8>& lhs,const Eigen::Matrix<double, 3, 8>& rhs, Eigen::Matrix3d* models) {
  Eigen::Matrix<double, 8, 9> A;

  for(int i = 0; i < 8; i++) {
    Eigen::Matrix<double, 1, 9> row;
    double x = lhs(0, i);
    double y = lhs(1, i);
    double x_dot = rhs(0, i);
    double y_dot = rhs(1, i);
    row << rhs(0, i) * lhs.col(i).transpose(),
        rhs(1, i) * lhs.col(i).transpose(), rhs(2, i) * lhs.col(i).transpose();
    row << x_dot * x, x_dot * y, x_dot,
           y_dot * x, y_dot * y, y_dot, 
           x, y, 1.0;
    A.row(i) = row;
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
  // std::printf("Minimum Singular Value : %f\n", singular(2));
  singular(2) = 0.0;
  *models = U * singular.asDiagonal() * V.transpose();
  // *models = U * singular.asDiagonal() * V;
  // std::cout << rhs.transpose() * F * lhs << std::endl;
}

void FitImpl(const Eigen::Matrix<double, 2, 8>& lhs,const Eigen::Matrix<double, 2, 8>& rhs, Eigen::Matrix3d* models) {
  Eigen::Matrix<double, 3, 8> lhs_homogous = lhs.colwise().homogeneous();
  Eigen::Matrix<double, 3, 8> rhs_homogous = rhs.colwise().homogeneous();

  Eigen::Matrix3d T = Normal(lhs);
  Eigen::Matrix3d T_dot = Normal(rhs);

  lhs_homogous = T * lhs_homogous;
  rhs_homogous = T_dot * rhs_homogous;
  // lhs^t * F * rhs = 0
  // (T * lhs)^t * F * (T_dot * rhs) = 0
  // lhs^t * T^t * T^-t * F * T_dot-1 * T_dot * rhs = 0 
  FitImpl(lhs_homogous, rhs_homogous, models);

  *models = T_dot.transpose() * (*models) * T;
}

void EightPointFundamentalSolverImpl::Fit(
    const std::vector<EightPointFundamentalSolverImpl::DataPointType,
                      Eigen::aligned_allocator<
                          EightPointFundamentalSolverImpl::DataPointType>>&
        data_points,
    Eigen::Matrix3d* models) {
  assert(data_points.size() == EightPointFundamentalSolverImpl::MINIMUM_DATA_POINT);
  Eigen::Matrix<double, 2, 8> lhs, rhs;
  for (int i = 0; i < 8; i++) {
    lhs.col(i) = data_points[i].first;
    rhs.col(i) = data_points[i].second;
  }
  FitImpl(lhs, rhs, models);
}
