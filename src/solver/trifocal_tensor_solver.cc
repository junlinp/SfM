#include "trifocal_tensor_solver.hpp"

#include <assert.h>

#include <iostream>

#include "algebra.hpp"
#include "ceres/autodiff_cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"
#include "eigen_alias_types.hpp"
#include "solver/bundle_adjustment.hpp"
#include "solver/triangular_solver.hpp"

double Error(TripleMatch data_point, Trifocal model) {
  Eigen::Vector3d x = data_point.I_observation.homogeneous();
  Eigen::Matrix3d lhs = SkewMatrix(data_point.J_observation.homogeneous());
  Eigen::Matrix3d rhs = SkewMatrix(data_point.K_observation.homogeneous());

  Eigen::Matrix3d tri_tensor =
      x(0) * model.lhs + x(1) * model.middle + x(2) * model.rhs;

  Eigen::Matrix3d result = lhs * tri_tensor * rhs;
  // std::cout << "tri_tensor : " << tri_tensor << std::endl;
  return std::sqrt((result.array().square()).sum());
}

std::ostream& operator<<(std::ostream& os, Trifocal tirfocal) {
  os << "Lhs : " << tirfocal.lhs << std::endl
     << "Middle : " << tirfocal.middle << std::endl
     << "Rhs : " << tirfocal.rhs << std::endl;
  return os;
}

double GeometryError(const TripleMatch data_point, Trifocal& model) {
  Mat34 P1, P2, P3;
  RecoveryCameraMatrix(model, P1, P2, P3);

  std::vector<Mat34> p_matrixs = {P1, P2, P3};
  std::vector<Eigen::Vector3d> obs = {data_point.I_observation.homogeneous(), data_point.J_observation.homogeneous(),
                                             data_point.K_observation.homogeneous()};

  Eigen::Vector4d X;
  BundleAdjustmentTriangular(p_matrixs, obs, X);
  double x[3] = {X(0) / X(3), X(1) / X(3), X(2) / X(3)};
  ceres::Problem problem;
  for (int i = 0; i < 3; i++) {
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<ConstCameraMatrixCostFunctor, 2, 3>(
            new ConstCameraMatrixCostFunctor(p_matrixs[i], obs[i]));
    problem.AddResidualBlock(cost_function, nullptr, x);
  }

  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
  double cost = 0.0;
  problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr,
                   nullptr);

  X << x[0], x[1], x[2], 1.0;
  Eigen::Vector3d p1 = P1 * X;
  p1 = p1 / p1(2);
  std::cout << p1 << std::endl;
  std::cout << data_point.I_observation << std::endl;
  std::cout << "p1 - v1 : " << (data_point.I_observation.homogeneous()- p1).norm() << std::endl;
  return cost;
}

void RecoveryCameraMatrix(const Trifocal& trifocal, Mat34& P1, Mat34& P2, Mat34& P3) {
  P1 << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;

  // T = [T_1, T_2, T_3]
  // let u_i be the left null vector of T_i
  // and v_i be the right null vector of T_i
  //
  // thus the epipolar of the second image - e'
  // (e')^T [u_1, u_2, u_3] = 0^T
  //

  // the epipolar of the third image - e''
  // (e'')^T [v_1, v_2, v_3] = 0^T
  //

  // P' = [ [T1, T2, T3] e'' | e']
  // P'' = [(e'' * e''^T - I)[T1^T, T2^T, T3^T] e' | e'']

  Eigen::Vector3d u1, u2, u3;
  Eigen::Vector3d v1, v2, v3;

  u1 = NullSpace(trifocal.lhs.transpose());
  u2 = NullSpace(trifocal.middle.transpose());
  u3 = NullSpace(trifocal.rhs.transpose());

  v1 = NullSpace(trifocal.lhs);
  v2 = NullSpace(trifocal.middle);
  v3 = NullSpace(trifocal.rhs);

  Eigen::Matrix3d temp;
  temp << u1.transpose(), u2.transpose(), u3.transpose();
  Eigen::Vector3d epipolar_ = NullSpace(temp);
  epipolar_.normalize();
  std::cout << "[u1, u2, u3] * e' : " << temp * epipolar_ << std::endl;

  temp << v1.transpose(), v2.transpose(), v3.transpose();
  Eigen::Vector3d epipolar_2 = NullSpace(temp);
  epipolar_2.normalize();
  // std::cout << "[v1, v2, v3] * e'' : " << temp * epipolar_2 << std::endl;

  Eigen::Matrix3d helper =
      epipolar_2 * epipolar_2.transpose() - Eigen::Matrix3d::Identity();

  P2 << trifocal.lhs * epipolar_2, trifocal.middle * epipolar_2,
      trifocal.rhs * epipolar_2, epipolar_;

  P3 << helper * trifocal.lhs.transpose() * epipolar_,
      helper * trifocal.middle.transpose() * epipolar_,
      helper * trifocal.rhs.transpose() * epipolar_, epipolar_2;

  Eigen::Vector3d b4 = P3.col(3);
  Eigen::Vector3d b3 = P3.col(2);
  Eigen::Vector3d b2 = P3.col(1);
  Eigen::Vector3d b1 = P3.col(0);

  Eigen::Vector3d a4 = P2.col(3);
  Eigen::Vector3d a3 = P2.col(2);
  Eigen::Vector3d a2 = P2.col(1);
  Eigen::Vector3d a1 = P2.col(0);

  Eigen::Matrix3d lhs = a1 * b4.transpose() - a4 * b1.transpose();
  ;
  Eigen::Matrix3d middle = a2 * b4.transpose() - a4 * b2.transpose();
  Eigen::Matrix3d rhs = a3 * b4.transpose() - a4 * b3.transpose();

  std::cout << "Recovery " << std::endl;
  std::cout << trifocal.lhs - lhs << std::endl << std::endl;
  std::cout << trifocal.middle - middle << std::endl << std::endl;
  std::cout << trifocal.rhs - rhs << std::endl << std::endl;
  std::cout << "Recovery end" << std::endl;
}

void BundleRecovertyCameraMatrix(const std::vector<TripleMatch>& data_points,
                                 const Trifocal& trifocal, const Mat34& P1,
                                 const Mat34& P2, Mat34& P3) {
  Mat34 P1_, P2_, P3_;

  RecoveryCameraMatrix(trifocal, P1_, P2_, P3_);
  Eigen::Matrix4d H;
  Eigen::Vector4d null_vector = NullSpace(P1);
  H << P1, null_vector.transpose();
  // TODO (junlinp@qq.com):
  // checkout whether P2_ = P2 * H
  double e = (P2 * H - P2_).array().square().sum();
  std::cout << "P2 * H == P2_ : " << e << std::endl;
  P3 = P3_ * H.inverse();

  std::vector<std::vector<double>> points(data_points.size(), std::vector<double>(3, 0));
  ceres::Problem probelm;
  for (int i = 0; i < data_points.size(); i++) {
      const TripleMatch& data_point = data_points[i]; 
      Eigen::Vector4d X;
      DLT({P1, P2, P3}, {data_point.I_observation.homogeneous(), data_point.J_observation.homogeneous(), data_point.K_observation.homogeneous()}, X);
      points[i][0] = X(0) / X(3);
      points[i][1] = X(1) / X(3);
      points[i][2] = X(2) / X(3);

      ceres::CostFunction* cost_function_p1 = new ceres::AutoDiffCostFunction<ConstCameraMatrixCostFunctor, 2, 3>(
          new ConstCameraMatrixCostFunctor(P1, data_point.I_observation.homogeneous())
      );
      probelm.AddResidualBlock(cost_function_p1, nullptr, points[i].data());

      ceres::CostFunction* cost_function_p2 = new ceres::AutoDiffCostFunction<ConstCameraMatrixCostFunctor, 2, 3>(
          new ConstCameraMatrixCostFunctor(P2, data_point.J_observation.homogeneous())
      );
      probelm.AddResidualBlock(cost_function_p2, nullptr, points[i].data());

      ceres::CostFunction* cost_function_p3 = new ceres::AutoDiffCostFunction<CostFunctor, 2, 12, 3>(
          new CostFunctor(data_point.K_observation.homogeneous())
      );

      probelm.AddResidualBlock(cost_function_p3, nullptr, P3.data(), points[i].data());
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 500;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &probelm, &summary);

  if (!summary.IsSolutionUsable()) {
      std::printf("Solution is unavailable\n");
  }
}
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
  // assert(data_points.size() == 7);
  size_t n = data_points.size();

  // Eigen::Matrix<double, 4 * n, 27> A;
  Eigen::MatrixXd A(4 * n, 27);
  A.setZero();
  int index = 0;
  for (const DataPointType& data_point : data_points) {
    Eigen::Vector3d x = data_point.I_observation.homogeneous();
    Eigen::Vector3d x_dot = data_point.J_observation.homogeneous();
    Eigen::Vector3d x_dot_dot = data_point.K_observation.homogeneous();
    Eigen::Matrix3d L = SkewMatrix(x_dot);
    Eigen::Matrix3d R = SkewMatrix(x_dot_dot);
    for (int r = 0; r < 2; r++) {
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
  assert(index == 4 * n);

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

void BundleRefineSolver::Fit(const std::vector<DataPointType>& data_points,
                             ModelType& model) {
  Mat34 P1, P2, P3;
  LinearSolver linear_solver;
  linear_solver.Fit(data_points, model);
  RecoveryCameraMatrix(model, P1, P2, P3);
  std::vector<std::vector<double>> points;
  std::vector<Mat34> p_matrixs = {P1, P2, P3};

  ceres::Problem problem;
  for (const DataPointType& data_point : data_points) {
    std::vector<Eigen::Vector3d> obs = {
        data_point.I_observation.homogeneous(), data_point.J_observation.homogeneous(), data_point.K_observation.homogeneous()};
    Eigen::Vector4d X;
    DLT(p_matrixs, obs, X);
    points.push_back({X(0) / X(3), X(1) / X(3), X(2) / X(3)});

    ceres::CostFunction* cost_function_p1 =
        new ceres::AutoDiffCostFunction<ConstCameraMatrixCostFunctor, 2, 3>(
            new ConstCameraMatrixCostFunctor(P1, data_point.I_observation.homogeneous()));
    problem.AddResidualBlock(cost_function_p1, nullptr, points.back().data());

    ceres::CostFunction* cost_function_p2 =
        new ceres::AutoDiffCostFunction<CostFunctor, 2, 12, 3>(
            new CostFunctor(data_point.J_observation.homogeneous()));
    problem.AddResidualBlock(cost_function_p2, nullptr, P2.data(),
                             points.back().data());
    ceres::CostFunction* cost_function_p3 =
        new ceres::AutoDiffCostFunction<CostFunctor, 2, 12, 3>(
            new CostFunctor(data_point.K_observation.homogeneous()));
    problem.AddResidualBlock(cost_function_p3, nullptr, P3.data(),
                             points.back().data());
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 500;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
  Eigen::Vector3d b4 = P3.col(3);
  Eigen::Vector3d b3 = P3.col(2);
  Eigen::Vector3d b2 = P3.col(1);
  Eigen::Vector3d b1 = P3.col(0);

  Eigen::Vector3d a4 = P2.col(3);
  Eigen::Vector3d a3 = P2.col(2);
  Eigen::Vector3d a2 = P2.col(1);
  Eigen::Vector3d a1 = P2.col(0);

  model.lhs = a1 * b4.transpose() - a4 * b1.transpose();
  ;
  model.middle = P2.col(1) * b4.transpose() - a4 * P3.col(1).transpose();
  model.rhs = P2.col(2) * b4.transpose() - a4 * P3.col(2).transpose();
}