#include "self_calibration_solver.hpp"
Eigen::Matrix<double, 1, 16> GenerateCoeffient(const Mat34 P, size_t row,
                                               size_t col) {
  Eigen::Matrix<double, 1, 16> res;
  Eigen::Vector4d rhs = P.row(col - 1);
  res << P(row - 1, 0) * rhs.transpose(), P(row - 1, 1) * rhs.transpose(),
      P(row - 1, 2) * rhs.transpose(), P(row - 1, 3) * rhs.transpose();
  return res;
}

Eigen::Matrix4d IAC(const std::vector<Mat34>& P, size_t image_width, size_t image_height) {
  size_t cx = image_width / 2;
  size_t cy = image_height / 2;

  // only use the pricipal point constraint

  //  4 * projective_reconstruction * 16
  size_t cameras_size = P.size();
  Eigen::MatrixXd coeffient(6 * cameras_size, 16);
  Eigen::VectorXd constant(6 * cameras_size);
  size_t count = 0;
  for (const auto& P_i : P) {
    coeffient.row(count * 6 + 0) = GenerateCoeffient(P_i, 1, 3);
    coeffient.row(count * 6 + 1) = GenerateCoeffient(P_i, 2, 3);
    coeffient.row(count * 6 + 2) = GenerateCoeffient(P_i, 3, 1);
    coeffient.row(count * 6 + 3) = GenerateCoeffient(P_i, 3, 2);
    coeffient.row(count * 6 + 4) = GenerateCoeffient(P_i, 1, 2);
    coeffient.row(count * 6 + 5) = GenerateCoeffient(P_i, 2, 1);
    constant(count * 6 + 0) = cx;
    constant(count * 6 + 1) = cy;
    constant(count * 6 + 2) = cx;
    constant(count * 6 + 3) = cy;
    constant(count * 6 + 4) = cy * cx;
    constant(count * 6 + 5) = cy * cx;
    count++;
  }

  // Solve Least-Squares Method
  Eigen::VectorXd Q_coeffient = coeffient.colPivHouseholderQr().solve(constant);
  Eigen::Matrix4d Q =
      Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(Q_coeffient.data());
  //
  // SVD Q = HIH with I is diag(1, 1, 1, 0)
  // But in pratice, Q may not has eigen vlaue with (1, 1, 1, 0) exactly
  // this solution of Q can be the initial value for the iterative methods.
  // Q can be decomposed as [ KK^T   -KK^Tp]
  //                        [ -p^TKK^T p^TKK^Tp]
  // Q will be parametered with K and p (K has 5 degree of freedom, and p has 3 degree of freedom)
  Eigen::JacobiSVD svd(Q, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector4d diag = svd.singularValues();
  diag(3) = 0.0;

  Q = svd.matrixU() * diag.asDiagonal() * svd.matrixV().transpose();
  return Q;
  
}


Eigen::Matrix3d RecoveryK(const Eigen::Matrix3d& dual_omega, size_t image_width, size_t image_height) {
    size_t cx = image_width / 2.0;
    size_t cy = image_height / 2.0;

    double fx = std::sqrt(dual_omega(0, 0) - cx * cx);
    double fy = std::sqrt(dual_omega(1, 1) - cy * cy);

    Eigen::Matrix3d K;
    K << fx, 0.0, cx,
         0.0, fy, cy,
         0.0, 0.0, 1.0;
    return K;
}