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

Eigen::Matrix3d SkewMatrix(const Eigen::Vector3d& v) {
    Eigen::Matrix3d res;
    res << 0.0, -v(2), v(0),
           v(2), 0.0,  -v(1),
           -v(0), v(1), 0.0;
    return res;
}
#endif  // SRC_SOLVER_ALGEBRA_HPP_