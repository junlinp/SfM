#include "epnp_solver.hpp"

#include "algebra.hpp"
#include "ceres/autodiff_cost_function.h"
#include "ceres/cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"

namespace solver {
void RecoveryRT(Eigen::Matrix<double, 3, 4> cc, Eigen::Matrix<double, 3, 4> cw,
                Mat33& R, Eigen::Vector3d& t) {
  using namespace Eigen;
  Vector3d cc_mean = cc.rowwise().mean();
  Vector3d cw_mean = cw.rowwise().mean();

  Matrix<double, 3, 4> cc_ = cc.colwise() - cc_mean;
  Matrix<double, 3, 4> cw_ = cw.colwise() - cw_mean;

  Mat33 H = cw_ * cc_.transpose();
  // std::cout << "H : " << H << std::endl;
  auto svd = H.bdcSvd(ComputeFullV | ComputeFullU);
  // std::cout << "RecoveryRT : " << svd.singularValues() << std::endl;
  Mat33 U = svd.matrixU();
  Mat33 V = svd.matrixV();
  double sign = (V * U.transpose()).determinant();

  Vector3d S(1.0, 1.0, sign > 0 ? 1.0 : -1.0);

  R = V * S.asDiagonal() * U.transpose();
  // std::cout << "Rotation : " << R << std::endl;
  t = cc_mean - R * cw_mean;
  // std::cout << "t : " << t << std::endl;
  // std::cout << "R * cw : " << R * cw << std::endl;
  // std::cout << "cc - R * cw - t : " << (cc - ((R * cw).colwise() + t))
  //          << std::endl;
}

Eigen::Matrix<double, 3, 4> ConvertCC(Eigen::VectorXd x) {
  Eigen::Matrix<double, 3, 4> res;
  res.col(0) = x.block(0, 0, 3, 1);
  res.col(1) = x.block(3, 0, 3, 1);
  res.col(2) = x.block(6, 0, 3, 1);
  res.col(3) = x.block(9, 0, 3, 1);
  return res;
}

double ReProjectError(double fx, double fy, double cx, double cy, Mat33 R,
                      Eigen::Vector3d t, Eigen::Vector3d X,
                      Eigen::Vector2d uv) {
  Eigen::Vector3d temp = R * X + t;
  temp /= temp(2);

  double res0 = fx * temp(0) + cx - uv(0);
  double res1 = fy * temp(1) + cy - uv(1);
  return (res0 * res0 + res1 * res1);
}

struct LMRefineFunctor {
  Eigen::VectorXd v0, v1, v2, v3;
  Eigen::MatrixXd cw;

  LMRefineFunctor(Eigen::VectorXd v0, Eigen::VectorXd v1, Eigen::VectorXd v2,
                  Eigen::VectorXd v3, Eigen::MatrixXd cw)
      : v0{v0}, v1{v1}, v2{v2}, v3{v3}, cw{cw} {}

  template <typename T>
  bool operator()(const T* beta, T* residual) const {
    T x[12];
    for (int i = 0; i < 12; i++) {
      x[i] =
          beta[0] * v0(i) + beta[1] * v1(i) + beta[2] * v2(i) + beta[3] * v3(i);
    }
    int count = 0;
    for (int i = 0; i < 4; i++) {
      for (int j = i + 1; j < 4; j++) {
        T cc_sub[3] = {x[3 * i + 0] - x[3 * j + 0], x[3 * i + 1] - x[3 * j + 1],
                       x[3 * i + 2] - x[3 * j + 2]};
        T res = cc_sub[0] * cc_sub[0] + cc_sub[1] * cc_sub[1] +
                cc_sub[2] * cc_sub[2] - (cw.col(i) - cw.col(j)).squaredNorm();
        residual[count++] = res;
      }
    }
    return true;
  }
};

bool BetaRefine_impl(Eigen::VectorXd v0, Eigen::VectorXd v1, Eigen::VectorXd v2,
                     Eigen::VectorXd v3, Eigen::MatrixXd cw, double* beta) {
  ceres::Problem problem;
  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<LMRefineFunctor, 6, 4>(
          new LMRefineFunctor(v0, v1, v2, v3, cw));
  problem.AddResidualBlock(cost_function, nullptr, beta);

  ceres::Solver::Options options;
  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);
  // std::cout << "BetaRefine : " << summary.BriefReport() << std::endl;
  return summary.IsSolutionUsable();
}

// biblibography
// EPnP: An Accurate O(n) Solution to the PnP Problem
bool EPnPSolver::Fit(const std::vector<DataPointType>& data_points,
                     Pose* pose) {
  size_t n = data_points.size();
  Eigen::MatrixXd pnp_2d(2, n);
  Eigen::MatrixXd pnp_3d(3, n);
  Eigen::MatrixXd pnp_3d_homo(4, n);

  for (int i = 0; i < n; i++) {
    pnp_2d.col(i) = data_points[i].first;
    pnp_3d.col(i) = data_points[i].second;
    pnp_3d_homo.col(i) = data_points[i].second.homogeneous();
  }
  Eigen::Vector3d pnp_3d_mean = pnp_3d.rowwise().mean();
  // std::cout << "c0 : " << pnp_3d_mean << std::endl;
  Eigen::MatrixXd centred_pnp_3d = pnp_3d.colwise() - pnp_3d_mean;
  auto svd = (centred_pnp_3d * centred_pnp_3d.transpose())
                 .bdcSvd(Eigen::ComputeFullV | Eigen::ComputeFullU);
  Eigen::Matrix3d V = svd.matrixV();
  Eigen::Vector3d S = svd.singularValues();
  // std::cout << "Singular Values : " << S << std::endl;
  // std::cout << "V : " << V << std::endl;
  Eigen::Vector3d c1 = pnp_3d_mean - S(0) * V.col(0);
  Eigen::Vector3d c2 = pnp_3d_mean - S(1) * V.col(1);
  Eigen::Vector3d c3 = pnp_3d_mean + S(2) * V.col(2);

  Eigen::Matrix<double, 3, 4> cw;
  cw << pnp_3d_mean, c1, c2, c3;
  Eigen::Matrix<double, 4, 4> cw_homo = cw.colwise().homogeneous();
  // std::cout << "cw_homo : " << cw_homo << std::endl;
  //  cw_homo * alpha = pnp_3d_homo
  Eigen::MatrixXd alpha = cw_homo.fullPivLu().solve(pnp_3d_homo);
  // std::cout << "cw * alpha == pnp_3d_homo : " << pnp_3d_homo - cw_homo *
  // alpha
  //           << std::endl;
  // std::cout << "alpha : " << alpha << std::endl;

  Eigen::Vector3d x_prod(1.0, 0.0, 0.0);
  Eigen::Vector3d y_prod(0.0, 1.0, 0.0);
  Eigen::Vector3d z_prod(0.0, 0.0, 1.0);

  double uc = cx, vc = cy;
  Eigen::MatrixXd M(2 * n, 12);
  for (int i = 0; i < n; i++) {
    double ui = pnp_2d(0, i);
    double vi = pnp_2d(1, i);
    M.row(2 * i + 0) = (fx * KrockerProduct(alpha.col(i), x_prod) +
                        (uc - ui) * KrockerProduct(alpha.col(i), z_prod))
                           .transpose();
    M.row(2 * i + 1) = (fy * KrockerProduct(alpha.col(i), y_prod) +
                        (vc - vi) * KrockerProduct(alpha.col(i), z_prod))
                           .transpose();
  }
  // std::cout << "M : " << M << std::endl;
  Eigen::Matrix<double, 12, 12> MTM = M.transpose() * M;
  // std::cout << "MTM : " << MTM << std::endl;
  auto M_svd = MTM.bdcSvd(Eigen::ComputeFullV);
  Eigen::VectorXd M_S = M_svd.singularValues();
  // std::cout << "M's Singular Values : " << M_S << std::endl;
  Eigen::MatrixXd M_V = M_svd.matrixV();

  Eigen::VectorXd v0 = -M_V.col(11);
  Eigen::VectorXd v1 = -M_V.col(10);
  Eigen::VectorXd v2 = -M_V.col(9);
  Eigen::VectorXd v3 = -M_V.col(8);
  // std::cout << "v0 : " << v0 << std::endl;
  // std::cout << "v1 : " << v1 << std::endl;
  // std::cout << "v2 : " << v2 << std::endl;
  // std::cout << "v3 : " << v3 << std::endl;
  std::vector<std::array<double, 4>> parameters;

  Eigen::VectorXd rho(6);
  int count = 0;
  for (int i = 0; i < 4; i++) {
    for (int j = i + 1; j < 4; j++) {
      Eigen::VectorXd temp = cw.col(i) - cw.col(j);
      rho(count++) = temp.dot(temp);
    }
  }
  // N = 1
  Eigen::MatrixXd N_A(6, 1);
  count = 0;
  for (int i = 0; i < 4; i++) {
    for (int j = i + 1; j < 4; j++) {
      Eigen::VectorXd temp =
          v0.block(3 * i, 0, 3, 1) - v0.block(3 * j, 0, 3, 1);
      N_A(count++, 0) = temp.dot(temp);
    }
  }
  Eigen::VectorXd beta_N_1 =
      N_A.bdcSvd(Eigen::ComputeFullV | Eigen::ComputeFullU).solve(rho);
  // std::cout << "beta_N_1 : " << beta_N_1 << std::endl;
  double N_1_beta1 = std::sqrt(beta_N_1(0));
  parameters.push_back({N_1_beta1, 0.0, 0.0, 0.0});
  parameters.push_back({-N_1_beta1, 0.0, 0.0, 0.0});

  Mat33 K;
  K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
  Eigen::MatrixXd cc = ConvertCC(N_1_beta1 * v0);
  // std::cout << "K * cc : " << (K * cc * alpha).colwise().hnormalized()
  //           << std::endl;
  // std::cout << "origin : " << pnp_2d << std::endl;
  //  N = 2

  Eigen::MatrixXd N_A_2(6, 3);
  count = 0;

  for (int i = 0; i < 4; i++) {
    for (int j = i + 1; j < 4; j++) {
      Eigen::VectorXd temp1 =
          v0.block(3 * i, 0, 3, 1) - v0.block(3 * j, 0, 3, 1);
      Eigen::VectorXd temp2 =
          v1.block(3 * i, 0, 3, 1) - v1.block(3 * j, 0, 3, 1);
      N_A_2(count, 0) = temp1.dot(temp1);
      N_A_2(count, 1) = temp1.dot(temp2) * 2;
      N_A_2(count, 2) = temp2.dot(temp2);
      count++;
    }
  }
  Eigen::VectorXd beta_N_2 =
      N_A_2.bdcSvd(Eigen::ComputeFullV | Eigen::ComputeFullU).solve(rho);
  beta_N_2 *= beta_N_2(0) > 0 ? 1.0 : -1.0;
  double N_2_beta_1 = std::sqrt(beta_N_2(0));
  double N_2_beta_2 = std::sqrt(beta_N_2(2));
  parameters.push_back({N_2_beta_1, N_2_beta_2, 0.0, 0.0});
  parameters.push_back({N_2_beta_1, -N_2_beta_2, 0.0, 0.0});
  parameters.push_back({N_2_beta_1, N_2_beta_2, 0.0, 0.0});
  parameters.push_back({-N_2_beta_1, -N_2_beta_2, 0.0, 0.0});


  // N = 3
  Eigen::MatrixXd N_A_3(6, 6);
  count = 0;
  for (int i = 0; i < 4; i++) {
    for (int j = i + 1; j < 4; j++) {
      Eigen::VectorXd temp1 =
          v0.block(3 * i, 0, 3, 1) - v0.block(3 * j, 0, 3, 1);
      Eigen::VectorXd temp2 =
          v1.block(3 * i, 0, 3, 1) - v1.block(3 * j, 0, 3, 1);
      Eigen::VectorXd temp3 =
          v2.block(3 * i, 0, 3, 1) - v2.block(3 * j, 0, 3, 1);
      N_A_3(count, 0) = temp1.dot(temp1);
      N_A_3(count, 1) = temp1.dot(temp2) * 2.0;
      N_A_3(count, 2) = temp1.dot(temp3) * 2.0;
      N_A_3(count, 3) = temp2.dot(temp2);
      N_A_3(count, 4) = temp2.dot(temp3) * 2.0;
      N_A_3(count, 5) = temp3.dot(temp3);
    }
  }
  Eigen::VectorXd beta_N_3 =
      N_A_3.bdcSvd(Eigen::ComputeFullV | Eigen::ComputeFullU).solve(rho);
  beta_N_3 *= beta_N_3(0) > 0 ? 1.0 : -1.0;
  double N_3_beta_1 = std::sqrt(beta_N_3(0));
  double N_3_beta_2 = std::sqrt(beta_N_3(3));
  double N_3_beta_3 = std::sqrt(beta_N_3(5));
  parameters.push_back({N_3_beta_1, N_3_beta_2, N_3_beta_3, 0.0});
  parameters.push_back({N_3_beta_1, N_3_beta_2, -N_3_beta_3, 0.0});
  parameters.push_back({N_3_beta_1, -N_3_beta_2, N_3_beta_3, 0.0});
  parameters.push_back({N_3_beta_1, -N_3_beta_2, -N_3_beta_3, 0.0});
  parameters.push_back({-N_3_beta_1, N_3_beta_2, N_3_beta_3, 0.0});
  parameters.push_back({-N_3_beta_1, N_3_beta_2, -N_3_beta_3, 0.0});
  parameters.push_back({-N_3_beta_1, -N_3_beta_2, N_3_beta_3, 0.0});
  parameters.push_back({-N_3_beta_1, -N_3_beta_2, -N_3_beta_3, 0.0});

  // N = 4
  // Not implement

  auto BetaRefine = [v0, v1, v2, v3, cw](double* beta) {
    return BetaRefine_impl(v0, v1, v2, v3, cw, beta);
  };
  // Compute the lowest error of Re-Project for four solution
  double error = std::numeric_limits<double>::max();
  for (auto& beta : parameters) {
    BetaRefine(beta.data());
    Eigen::VectorXd x =
        beta[0] * v0 + beta[1] * v1 + beta[2] * v2 + beta[3] * v3;
    Mat33 R;
    Eigen::Vector3d t;
    RecoveryRT(ConvertCC(x), cw, R, t);
    double e = 0.0;
    for (int i = 0; i < n; i++) {
      e += ReProjectError(fx, fy, cx, cy, R, t, pnp_3d.col(i), pnp_2d.col(i));
    }
    e /= n;
    if (e < error) {
        error = e;
      *pose = Pose(R, -R.transpose() * t);
    }
  }
  
  std::cout << "EPnP Minimum Error : " << error << std::endl;

  return true;
}
}  // namespace solver