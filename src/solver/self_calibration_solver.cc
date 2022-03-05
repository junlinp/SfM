#include "self_calibration_solver.hpp"
#include "ceres/problem.h"
#include "ceres/solver.h"
#include "ceres/autodiff_cost_function.h"
#include "ceres/loss_function.h"

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

struct Cost_Functor {
  Mat34 P;
  Cost_Functor(Mat34 P) : P(P){}

  template<typename T>
  bool operator()(const T* parameter, T* output) const {
    Eigen::Matrix<T, 3, 3> K;
    K << parameter[0], T(0.0), parameter[2],
         T(0.0), parameter[1], parameter[3],
         T(0.0), T(0.0), T(1.0);
    Eigen::Matrix<T, 3, 1> p(parameter[4], parameter[5], parameter[6]);

    Eigen::Matrix<T, 4, 4> Q;
    Q << K * K.transpose(), -K*K.transpose()*p,
         -p.transpose() * K * K.transpose(), p.transpose() * K * K.transpose() *p;

    Eigen::Matrix<T, 3, 3> temp = K * K.transpose() - P * Q * P.transpose();

    std::copy(temp.data(), temp.data() + 9, output);
    return true;
  }
};

struct QuadrticFunctor {
  Mat34 P1, P2;
  QuadrticFunctor(Mat34 P1, Mat34 P2) : P1(P1), P2(P2) {}

  template<typename T>
  bool operator()(const T* parameter, T* output) const {
    Eigen::Matrix<T, 3, 3> K;
    K << parameter[0], T(0.0), parameter[2],
         T(0.0), parameter[1], parameter[3],
         T(0.0), T(0.0), T(1.0);
    Eigen::Matrix<T, 3, 1> p(parameter[4], parameter[5], parameter[6]);

    Eigen::Matrix<T, 4, 4> Q;
    Q << K * K.transpose(), -K*K.transpose()*p,
         -p.transpose() * K * K.transpose(), p.transpose() * K * K.transpose() *p;
   
    Eigen::Matrix<T, 3, 3> omega1, omega2;
    omega1 = K * K.transpose();
    omega2 = P2 * Q * P2.transpose();

    output[0] = omega1(0, 0) * omega2(0, 1) - omega2(0, 0) * omega1(0, 1);
    output[1] = omega1(0, 1) * omega2(0, 2) - omega2(0, 1) * omega1(0, 2);
    output[2] = omega1(0, 2) * omega2(1,1) - omega2(0, 2) * omega1(1, 1);
    output[3] = omega1(1, 1) * omega2(1, 2) - omega2(1, 1) * omega1(1, 2);
    output[4] = omega1(1, 2) * omega2(2, 2) - omega2(1, 2) * omega1(2, 2);
    return true;
  }
};

struct AbsoluteFunctor {
  Mat34 P;
  double cx, cy;
  AbsoluteFunctor(Mat34 p, double cx, double cy) : P(p), cx(cx), cy(cy){}

  template<typename T>
  bool operator()(const T* parameter, T* output) const {

    Eigen::Matrix<T, 3, 3> K;
    K << parameter[0], T(0.0), T(cx),
         T(0.0), parameter[1], T(cy),
         T(0.0), T(0.0), T(1.0);
    Eigen::Matrix<T, 3, 1> p(parameter[2], parameter[3], parameter[4]);

    Eigen::Matrix<T, 3, 3> M; 
    Eigen::Matrix<T, 3, 1> a;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        M(i, j) = T(P(i, j));
      }
      a(i, 0) = T(P(i, 3));
    }
    
    Eigen::Matrix<T, 3, 3> t = M - a * p.transpose();
    // det(M + a * pT)
    T lambda = t(0, 0) * (t(1, 1) * t(2, 2) - t(2, 1) * t(1, 2)) - t(0, 1) * (t(1, 0) * t(2, 2) - t(2, 0) * t(1, 2)) + t(0, 2) * (t(1, 0) * t(2, 1) - t(2, 0) * t(1, 1));
    T determinant_squared = lambda * lambda;
    Eigen::Matrix<T, 3, 3> result_matrix = determinant_squared * K * K.transpose() - t * K * K.transpose() * t.transpose();

    output[0] = result_matrix(0, 0);
    output[1] = result_matrix(0, 1);
    output[2] = result_matrix(0, 2);
    output[3]= result_matrix(1, 1);
    output[4] = result_matrix(1, 2);
    return true;
  }
};


template<typename Functor, int kNumResiduals, int kNumParameters, typename... Args>
ceres::CostFunction* MinimumCostFunction(Args... args) {
  return new ceres::AutoDiffCostFunction<Functor, kNumResiduals, kNumParameters>(new Functor(args...));
}

// Bibliography
// <Self-Calibration from the Absolute Conic on the Plane at Infinity>
// Marc Pollefeys and Luc Van Gool
bool IterativeSolver::Solve(std::vector<Mat34>& Ps) {

  ceres::Problem problem;
 for (int i = 1; i < Ps.size(); i++) {
   auto cost_function = MinimumCostFunction<AbsoluteFunctor, 5, 5>(Ps[i], cx, cy);
   ceres::LossFunction* loss_function = new ceres::HuberLoss(4.0);
   problem.AddResidualBlock(cost_function, loss_function, parameters);
 }

  ceres::Solver::Options options;
  options.max_num_iterations = 1024 * 16;
  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  parameters[0] = std::abs(parameters[0]);
  parameters[1] = std::abs(parameters[1]);

  //ceres::Solve(options, &problem, &summary);
  //std::cout << summary.BriefReport() << std::endl;

  return summary.IsSolutionUsable();
  
}

Mat33 IterativeSolver::K() {
  Mat33 K;
  K << parameters[0], 0.0, cx,
       0.0, parameters[1], cy,
       0.0, 0.0, 1.0;
  return K;
}

Eigen::Vector3d IterativeSolver::p() {
  return Eigen::Vector3d(parameters[2], parameters[3], parameters[4]);
}

Eigen::Matrix4d IterativeSolver::HomogeneousMatrix() {
  Eigen::Matrix4d res;
  Eigen::Vector3d zero;
  zero.setZero();
  res << K(), zero,
        -p().transpose() * K(), 1;
  return res;
}
