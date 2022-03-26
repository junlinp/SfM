#include "trifocal_tensor_solver.hpp"

#include <assert.h>

#include <iostream>

#include "algebra.hpp"
#include "ceres/autodiff_cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"
#include "ceres/jet.h"
#include "eigen_alias_types.hpp"
#include "solver/bundle_adjustment.hpp"
#include "solver/triangular_solver.hpp"
#include "algebra.hpp"
/*
double TrifocalError::Error(TripleMatch data_point, Trifocal model) {
  Eigen::Vector3d x = data_point.I_observation.homogeneous();
  Eigen::Matrix3d lhs = SkewMatrix(data_point.J_observation.homogeneous());
  Eigen::Matrix3d rhs = SkewMatrix(data_point.K_observation.homogeneous());

  Eigen::Matrix3d tri_tensor =
   x(0) * model.lhs + x(1) * model.middle + x(2) * model.rhs;

  Eigen::Matrix3d zeros = lhs * tri_tensor * rhs;
  return (zeros.array().square()).sum();
}

bool TrifocalError::RejectRegion(double error) {
  // how to define the error
  // chi_square(9)
  return error * 0.5 * 0.5 > 16.92;
}
*/
double TrifocalReprojectiveErrorEvaluator::Error(TripleMatch data_point, Trifocal model) {
  Mat34 P1, P2, P3;
  RecoveryCameraMatrix(model, P1, P2, P3);
    Eigen::Vector4d X;
  DLT({P1, P2, P3}, {data_point.I_observation.homogeneous(), data_point.J_observation.homogeneous(), data_point.K_observation.homogeneous()}, X);
  ceres::Problem problem;
  double point[3] = {X(0)/X(3), X(1)/X(3), X(2)/X(3)};
  ceres::CostFunction* cost_function_p1 =
      new ceres::AutoDiffCostFunction<ConstCameraMatrixCostFunctor, 2, 3>(
          new ConstCameraMatrixCostFunctor(P1, data_point.I_observation));
  ceres::CostFunction* cost_function_p2 =
      new ceres::AutoDiffCostFunction<ConstCameraMatrixCostFunctor, 2, 3>(
          new ConstCameraMatrixCostFunctor(P2, data_point.J_observation));
  ceres::CostFunction* cost_function_p3 =
      new ceres::AutoDiffCostFunction<ConstCameraMatrixCostFunctor, 2, 3>(
          new ConstCameraMatrixCostFunctor(P3, data_point.K_observation));

  problem.AddResidualBlock(cost_function_p1, nullptr, point);
  problem.AddResidualBlock(cost_function_p2, nullptr, point);
  problem.AddResidualBlock(cost_function_p3, nullptr, point);

  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  
  ceres::Solve(options, &problem, &summary);
  // std::cout << "Trifocal Reprojective Error : " << std::endl << summary.BriefReport() << std::endl;
  return summary.final_cost * 2.0;
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
  //std::cout << summary.BriefReport() << std::endl;
  double cost = 0.0;
  problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr,
                   nullptr);

  X << x[0], x[1], x[2], 1.0;
  Eigen::Vector3d p1 = P1 * X;
  p1 = p1 / p1(2);
  //std::cout << p1 << std::endl;
  //std::cout << data_point.I_observation << std::endl;
  //std::cout << "p1 - v1 : " << (data_point.I_observation.homogeneous()- p1).norm() << std::endl;
  return cost;
}

template<typename T>
auto Inference(const Trifocal& model, T* lhs, T* middle, T* rhs, T* output) {
  T lhs_[3] = {T(lhs[0]), T(lhs[1]), T(1.0)};
  T middle_[3] = {T(middle[0]), T(middle[1]), T(1.0)};
  T rhs_[3] = {T(rhs[0]), T(rhs[1]), T(1.0)};
  
  auto middle_skew_matrix = SkewMatrix(middle_);
  auto rhs_skew_matrix = SkewMatrix(rhs_);

  using TMat33 = Eigen::Matrix<T, 3, 3, Eigen::RowMajor>; 
  TMat33 first, second, third;
  for (int i = 0 ; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      first(i, j) = T(model.lhs(i, j));
      second(i, j) = T(model.middle(i, j));
      third(i, j) = T(model.rhs(i, j));
    }
  }
  auto temp = middle_skew_matrix * (lhs[0] * first + lhs[1] * second + T(1.0) * third) * rhs_skew_matrix;

  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      output[row * 3 + col] = temp(row, col);
    }
  }
}

Eigen::VectorXd TrifocalSamponErrorEvaluator::Evaluate(const TripleMatch& data_point, const Trifocal& model) {
  
  double res[9] = {0.0};
  double data[6];
  data[0] = data_point.I_observation(0);
  data[1] = data_point.I_observation(1);
  data[2] = data_point.J_observation(0);
  data[3] = data_point.J_observation(1);
  data[4] = data_point.K_observation(0);
  data[5] = data_point.K_observation(1);
  Inference(model, &data[0], &data[2], &data[4], &res[0]);
  return Eigen::Map<Eigen::Matrix<double, 9, 1>>(res);
}

  Eigen::Matrix<double, 9, 6> TrifocalSamponErrorEvaluator::Jacobian(const TripleMatch& data_point, const Trifocal& model) {
    Eigen::Matrix<double, 9, 6> res;

    using Jet = ceres::Jet<double, 6>;
    ceres::Jet<double, 6> data[6];
    data[0] = Jet(data_point.I_observation(0), 0);
    data[1] = Jet(data_point.I_observation(1), 1);
    data[2] = Jet(data_point.J_observation(0), 2);
    data[3] = Jet(data_point.J_observation(1), 3);
    data[4] = Jet(data_point.K_observation(0), 4);
    data[5] = Jet(data_point.K_observation(1), 5);
    Jet output[9];

    Inference(model, &data[0], &data[2], &data[4], &output[0]);
    for(int i = 0; i < 9; i++) {
      res.row(i) = output[i].v;
    }
    return res;
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
  std::cout << "recovery Camera Matrix : " << std::endl << trifocal << std::endl;

  Eigen::Vector3d u1, u2, u3;
  Eigen::Vector3d v1, v2, v3;

  u1 = NullSpace(trifocal.lhs.transpose());
  u2 = NullSpace(trifocal.middle.transpose());
  u3 = NullSpace(trifocal.rhs.transpose());
  std::cout << "u1 : " << u1 << std::endl;
  std::cout << "u2 : " << u2 << std::endl;
  std::cout << "u3 : " << u3 << std::endl;

  v1 = NullSpace(trifocal.lhs);
  v2 = NullSpace(trifocal.middle);
  v3 = NullSpace(trifocal.rhs);

  Eigen::Matrix3d temp;
  temp.row(0) = u1.transpose();
  temp.row(1) = u2.transpose();
  temp.row(2) = u3.transpose();
  Eigen::Vector3d epipolar_ = NullSpace(temp);
  std::cout << "e1 : " << epipolar_ << std::endl;

  epipolar_.normalize();
  //std::cout << "[u1, u2, u3] * e' : " << temp * epipolar_ << std::endl;

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
  Eigen::Matrix3d middle = a2 * b4.transpose() - a4 * b2.transpose();
  Eigen::Matrix3d rhs = a3 * b4.transpose() - a4 * b3.transpose();
  /*
  std::cout << "Recovery " << std::endl;
  std::cout << trifocal.lhs - lhs << std::endl << std::endl;
  std::cout << trifocal.middle - middle << std::endl << std::endl;
  std::cout << trifocal.rhs - rhs << std::endl << std::endl;
  std::cout << "Recovery end" << std::endl;
  */
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
          new ConstCameraMatrixCostFunctor(P1, data_point.I_observation)
      );
      probelm.AddResidualBlock(cost_function_p1, nullptr, points[i].data());

      ceres::CostFunction* cost_function_p2 = new ceres::AutoDiffCostFunction<ConstCameraMatrixCostFunctor, 2, 3>(
          new ConstCameraMatrixCostFunctor(P2, data_point.J_observation)
      );
      probelm.AddResidualBlock(cost_function_p2, nullptr, points[i].data());

      ceres::CostFunction* cost_function_p3 = new ceres::AutoDiffCostFunction<CostFunctor, 2, 12, 3>(
          new CostFunctor(data_point.K_observation)
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

Eigen::VectorXd KroneckerProduct(Eigen::VectorXd lhs, Eigen::VectorXd rhs) {
  size_t n = lhs.rows();
  size_t m = rhs.rows();
  Eigen::VectorXd res(m * n);

  for (int i = 0; i < n; i++) {
    res.block(i * m, 0 , m, 1) = lhs(i) * rhs;
  }
  return res;
}

Eigen::Matrix<double, 27, 1> Trilinear(Eigen::Vector3d lhs, Eigen::Vector3d middle, Eigen::Vector3d rhs) {
  Eigen::Matrix<double, 9, 1> temp = KroneckerProduct(lhs, rhs);
  Eigen::Matrix<double, 27, 1> res = KroneckerProduct(middle, temp);
  return res;
}
template<typename DataPointType>
void ConstructCoeffientMatrix(const std::vector<DataPointType>& data_points, Eigen::MatrixXd& coeffient) {
  int index = 0;
  for (const DataPointType& data_point : data_points) {
    Eigen::Vector3d x = data_point.I_observation.homogeneous();
    Eigen::Vector3d x_dot = data_point.J_observation.homogeneous();
    Eigen::Vector3d x_dot_dot = data_point.K_observation.homogeneous();
    Eigen::Matrix3d L = SkewMatrix(x_dot);
    Eigen::Matrix3d R = SkewMatrix(x_dot_dot);

    coeffient.row(4 * index + 0) = Trilinear(L.row(0), x, R.col(0)).transpose();
    coeffient.row(4 * index + 1) = Trilinear(L.row(0), x, R.col(1)).transpose();
    coeffient.row(4 * index + 2) = Trilinear(L.row(1), x, R.col(0)).transpose();
    coeffient.row(4 * index + 3) = Trilinear(L.row(1), x, R.col(1)).transpose();
    index += 1;
  }
 }
template<typename T>
T Normal(const T* data, size_t size) {
  T res = T(0.0);
  for (size_t i = 0; i < size; i++) {
    res += data[i] * data[i];
  }
  return res;
}
 struct LinearFunctor {
   Eigen::Vector2d ob_x, ob_x_dot, ob_x_dot_dot;
   LinearFunctor(Eigen::Vector2d x, Eigen::Vector2d x_dot, Eigen::Vector2d x_dot_dot) : ob_x(x),   ob_x_dot{x_dot}, ob_x_dot_dot{x_dot_dot} {};
   template<typename T>
   bool operator()(const T* parameter, T* output) const {
     using TMatrix = Eigen::Matrix<T, 3, 3, Eigen::RowMajor>;
     TMatrix middle;
     for (int row = 0; row < 3; row++) {
       for (int col = 0; col < 3; col++) {
         middle(row, col) = T(ob_x.x()) * parameter[0 + row * 3 + col] + T(ob_x.y()) * parameter[9 + row * 3 + col]
         + parameter[18 + row * 3 + col];
       }
     }
     
     T lhs_data[3] = {T(ob_x_dot.x()), T(ob_x_dot.y()), T(1.0)};
     T rhs_data[3] = {T(ob_x_dot_dot.x()), T(ob_x_dot_dot.y()), T(1.0)};

     TMatrix lhs = SkewMatrix(lhs_data);
     TMatrix rhs = SkewMatrix(rhs_data);
     TMatrix res = lhs * middle * rhs;
     for(int row = 0; row < 3; row++) {
       for (int col = 0; col < 3; col++) {
         output[row * 3 + col] = res(row, col);
       }
     }
     output[9] = 1.0 - Normal(parameter, 27);
     return true;
   }
 };
void LinearSolver::Fit(const std::vector<DataPointType>& data_points,
                       ModelType* model) {
  // assert(data_points.size() == 7);
  size_t n = data_points.size();

  // T_i^(jk) expand to a vector(i * 9 + j * 3 + k)
  Eigen::MatrixXd A(4 * n, 27);
  A.setZero();
  ConstructCoeffientMatrix(data_points, A);

  Eigen::Matrix<double, 27, 1> t;
  LinearEquationWithNormalSolver(A, t);
  // convert t to model;
  
  double parameters[27] = {0.0};
  std::copy(t.data(), t.data() + 27, parameters);

  model->lhs =
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(parameters);
  model->middle =
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(parameters + 9);
  model->rhs =
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(parameters + 18);
}


void TrifocalVMatrix(Trifocal tri_focal, Mat33 & V) {
  Eigen::Vector3d v1, v2, v3;
  LinearEquationWithNormalSolver(tri_focal.lhs, v1);
  LinearEquationWithNormalSolver(tri_focal.middle, v2);
  LinearEquationWithNormalSolver(tri_focal.rhs, v3);
  V << v1.transpose(), v2.transpose(), v3.transpose();
}

// bibliography
// Multiple View Geometry in Compute Vision algorithm 16.2
void AlgebraMinimumSolver::Fit(const std::vector<DataPointType>& data_points, ModelType* model) {
  // there is a initialized value for estimate a solution menting the inner constrainted of trifocal
  Mat33 V, V_transpose;
  TrifocalVMatrix(*model, V);
  ModelType model_transpose;
  model_transpose.lhs = model->lhs.transpose();
  model_transpose.middle = model->middle.transpose();
  model_transpose.rhs = model->rhs.transpose();
  TrifocalVMatrix(model_transpose, V_transpose);

  Eigen::Vector3d epipolar_middle, epipolar_rhs;
  LinearEquationWithNormalSolver(V, epipolar_rhs);
  LinearEquationWithNormalSolver(V_transpose, epipolar_middle);

  Eigen::Matrix<double, 27, 18> E;
  E.setZero();

  // a = [ a_1, a_2, a_3, b_1, b_2, b_3]
  // 

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j ++) {
      for (int k = 0; k < 3; k++) {
        E(i * 9 + j * 3 + k, i * 3 + j) = epipolar_rhs(k);
        E(i * 9 + j * 3 + k, 9 + i * 3 + j) = epipolar_middle(j);
      }
    }
  }
  size_t n = data_points.size();
  Eigen::MatrixXd A(4 * n, 27);
  ConstructCoeffientMatrix(data_points, A);

  // Minimum |AEa| with |Ea| = 1
  // let x = Ea
  // TODO (junlinp@qq.com): it can simplify by hand.

  auto svd_E = E.bdcSvd(Eigen::ComputeFullU| Eigen::ComputeFullV);
  Eigen::Matrix<double, 27, 27> U_E = svd_E.matrixU();
  Eigen::Matrix<double, 27, 18> D_E;
  D_E.setZero();
  D_E.block(0, 0, 18, 18) = svd_E.singularValues().asDiagonal();

  auto V_E = svd_E.matrixV();

  Eigen::Matrix<double, 27, 27> V_T_A = A.bdcSvd(Eigen::ComputeFullV).matrixV().transpose();
  Eigen::Matrix<double, 27, 27> R = V_T_A * U_E;
  // the last row of R and the last col of R transpose
  // we have (0, 0, ...., 0, 1) ^T = R^T * x_dot
  Eigen::VectorXd x_dot = R.row(26);

  Eigen::Matrix<double, 18, 1> a = V_E * D_E.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(x_dot);

  Eigen::Matrix<double, 27, 1> t = E * a;

  using RowMajorMatrix = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
  using MapRowMajorMatrix = Eigen::Map<RowMajorMatrix>;
  model->lhs = MapRowMajorMatrix(t.data());
  model->middle = MapRowMajorMatrix(t.data() + 9);
  model->rhs = MapRowMajorMatrix(t.data() + 18);
  // we will not iterate this solution again
  // because we will do this process at the later step.
}

void BundleRefineSolver::Fit(const std::vector<DataPointType>& data_points,
                             ModelType* model) {
  Mat34 P1, P2, P3;

  NormalizedLinearSolver::Fit(data_points, model);

  RecoveryCameraMatrix(*model, P1, P2, P3);

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
            new ConstCameraMatrixCostFunctor(P1, data_point.I_observation));
    problem.AddResidualBlock(cost_function_p1, nullptr, points.back().data());

    ceres::CostFunction* cost_function_p2 =
        new ceres::AutoDiffCostFunction<CostFunctor, 2, 12, 3>(
            new CostFunctor(data_point.J_observation));
    problem.AddResidualBlock(cost_function_p2, nullptr, P2.data(),
                             points.back().data());
    ceres::CostFunction* cost_function_p3 =
        new ceres::AutoDiffCostFunction<CostFunctor, 2, 12, 3>(
            new CostFunctor(data_point.K_observation));
    problem.AddResidualBlock(cost_function_p3, nullptr, P3.data(),
                             points.back().data());
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 1024;
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

  model->lhs = a1 * b4.transpose() - a4 * b1.transpose();
  model->middle = a2 * b4.transpose() - a4 * b2.transpose();
  model->rhs = a3 * b4.transpose() - a4 * b3.transpose();

  // checkout the Reprojective Error
}