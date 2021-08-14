#ifndef SRC_SOLVER_ALGEBRA_HPP_
#define SRC_SOLVER_ALGEBRA_HPP_
template<typename Derived>
Eigen::VectorXd NullSpace(const Derived& m) {
  size_t col = m.cols();

  Eigen::JacobiSVD svd(m, Eigen::ComputeFullV);

  return svd.matrixV().row(col - 1);
}

Eigen::Vector3d NullSpace(const Eigen::Matrix3d& m) {
  Eigen::JacobiSVD svd(m, Eigen::ComputeFullV);
  return svd.matrixV().col(2);
}
#endif  // SRC_SOLVER_ALGEBRA_HPP_