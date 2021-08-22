#include "trifocal_tensor_solver.hpp"

#include <assert.h>
#include "algebra.hpp"

#include <iostream>

double Error(TriPair data_point, Trifocal model) {
  Eigen::Vector3d x = data_point.lhs;
  Eigen::Matrix3d lhs = SkewMatrix(data_point.middle);
  Eigen::Matrix3d rhs = SkewMatrix(data_point.rhs);

  Eigen::Matrix3d tri_tensor =
      x(0) * model.lhs + x(1) * model.middle + x(2) * model.rhs;

  Eigen::Matrix3d result = lhs * tri_tensor * rhs;

  return std::sqrt((result.array().square()).sum());
}

std::ostream& operator<<(std::ostream& os, Trifocal tirfocal) {
  os << "Lhs : " << tirfocal.lhs << std::endl
     << "Middle : " << tirfocal.middle << std::endl
     << "Rhs : " << tirfocal.rhs << std::endl;
     return os;
}

/*
void RecoveryCameraMatrix(Trifocal& trifocal, Mat34& P1, Mat34& P2, Mat34& P3) {
    P1 << 1.0, 0.0, 0.0, 0.0,
          0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0;

}
*/

namespace {
struct EpsilonTensor {
  static double Get(int i, int j, int k) {
    int v = i * 100 + j * 10 + k;

    switch (v) {
      case 123:
      case 132:
      case 321:
        return 1.0;
      case 312:
      case 213:
      case 231:
        return -1.0;
      default:
        return 0.0;
    }
    return 0.0;
  }
};
}  // namespace
void LinearSolver::Fit(const std::vector<DataPointType>& data_points,
                       ModelType& model) {
  assert(data_points.size() == 7);

  Eigen::Matrix<double, 4 * 7, 27> A;
  A.setZero();
  int index = 0;
  for (const DataPointType& data_point : data_points) {
    Eigen::Vector3d x = data_point.lhs;
    Eigen::Vector3d x_dot = data_point.middle;
    Eigen::Vector3d x_dot_dot = data_point.rhs;
    Eigen::Matrix3d L = SkewMatrix(x_dot);
    Eigen::Matrix3d R = SkewMatrix(x_dot_dot);
    for(int r = 0; r < 2; r++) {
        for (int s = 0; s < 2; s++) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        int cur = i * 9 + j * 3 + k;
                        A(index, cur) = L(r, j) * R(k, s) * x(i);
                    }
                }
            }
            index++;
        }
    }
  }
  assert(index == 28);

  Eigen::Matrix<double, 27, 1> t;
  LinearEquationWithNormalSolver(A, t);
  // convert t to model;
  model.lhs =
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(t.data());
  model.middle =
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(t.data() + 9);
  model.rhs =
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(t.data() + 18);
}
