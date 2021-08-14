#ifndef SRC_SOLVER_TRIANGULAR_SOLVER_HPP_
#define SRC_SOLVER_TRIANGULAR_SOLVER_HPP_

#include <assert.h>

#include <vector>

#include "Eigen/Dense"

using Mat34 = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;

void DLT(const std::vector<Mat34>& p_matrixs,
         const std::vector<Eigen::Vector3d>& obs, Eigen::Vector4d& X) {
  assert(p_matrixs.size() == obs.size());
  size_t size = p_matrixs.size();
  Eigen::MatrixXd coefficent(2 * size, 4);
  Eigen::VectorXd zeros_constant(2 * size);
  zeros_constant.setZero();

  for (int i = 0; i < size; i++) {
    const Mat34& P = p_matrixs.at(i);
    const Eigen::Vector3d& ob = obs.at(i);
    double u = ob.x();
    double v = ob.y();
    Eigen::Vector4d one, two;
    one << u * P.row(2) - P.row(1);
    two << P.row(0) - v * P.row(1);
    coefficent.row(2 * i + 0) = one;
    coefficent.row(2 * i + 1) = two;
  }
  // std::cout << "DLT coefficnet : " << coefficent << std::endl;
  Eigen::JacobiSVD svd(coefficent, Eigen::ComputeFullV);
  //std::cout << "Singular Value : " << svd.singularValues() << std::endl;
  //std::cout << "Zero Value : " << svd.matrixV().row(3) << std::endl;

  //X = coefficent.colPivHouseholderQr().solve(zeros_constant);
  X = svd.matrixV().row(3);
  // std::cout << "X : " << X << std::endl;
}
#endif  // SRC_SOLVER_TRIANGULAR_SOLVER_HPP_