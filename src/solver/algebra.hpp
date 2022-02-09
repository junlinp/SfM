#ifndef SRC_SOLVER_ALGEBRA_HPP_
#define SRC_SOLVER_ALGEBRA_HPP_

#include <iostream>
#include "Eigen/Dense"

template<typename Derived>
Eigen::VectorXd NullSpace(const Derived& m) {
  size_t col = m.cols();

  Eigen::JacobiSVD svd(m, Eigen::ComputeFullV);

  return svd.matrixV().col(col - 1);
}

template<typename T>
Eigen::VectorXd NullSpace(const Eigen::Transpose<T>& m) {
    T temp = m;
    return NullSpace<T>(temp);
}

//Eigen::Vector3d NullSpace(const Eigen::Matrix3d& m) {
//  Eigen::JacobiSVD svd(m, Eigen::ComputeFullV);
//  return svd.matrixV().col(2);
//}
template<class T>
Eigen::Matrix3d SkewMatrix(const T& v) {
    Eigen::Matrix3d res;
    res << 0.0, -v(2), v(1),
           v(2), 0.0,  -v(0),
           -v(1), v(0), 0.0;
    return res;
}

template<typename DerivedMatrix, typename DerivedVector>
void LinearEquationWithNormalSolver(const DerivedMatrix& A, DerivedVector& t) {
    Eigen::JacobiSVD svd(A, Eigen::ComputeFullV);
    auto singular = svd.singularValues();
    std::cout << "Least Singular : " << singular(singular.size() - 1) << std::endl;
    t = svd.matrixV().col(singular.size() - 1);
}
#endif  // SRC_SOLVER_ALGEBRA_HPP_