#include "triangular_solver.hpp"
#include "ceres/problem.h"
#include "ceres/autodiff_cost_function.h"
#include "ceres/solver.h"
#include "solver/bundle_adjustment.hpp"
#include "algebra.hpp"

#include <assert.h>

namespace {

double TriangularGeometryError(const Mat34& p, const Eigen::Vector3d& obs, const Eigen::Vector4d& X) {
    Eigen::Vector3d uv = p * X;
    return (obs.hnormalized() - uv.hnormalized()).squaredNorm();
}

double TriangularGeometryError(const std::vector<Mat34>& p_matrixs, const std::vector<Eigen::Vector3d>& obs, const Eigen::Vector4d& X) {
    double error = 0.0;
    assert(p_matrixs.size() == obs.size());
    size_t size = p_matrixs.size();
    for (int i = 0; i < size; i++) {
        error += TriangularGeometryError(p_matrixs[i], obs[i], X);
    }
    return error;

}

}

double DLT(const std::vector<Mat34>& p_matrixs,
         const std::vector<Eigen::Vector3d>& obs, Eigen::Vector4d& X) {
  assert(p_matrixs.size() == obs.size());
  size_t size = p_matrixs.size();
  Eigen::MatrixXd coefficent(2 * size, 4);

  for (int i = 0; i < size; i++) {
    const Mat34& P = p_matrixs.at(i);
    const Eigen::Vector3d& ob = obs.at(i);

    coefficent.row(2 * i) = P.row(0) - ob.x() * P.row(2);
    coefficent.row(2 * i + 1) = P.row(1) - ob.y() * P.row(2);
  }

  X = NullSpace(coefficent);

  //std::cout << (coefficent * X).squaredNorm() << std::endl;
  //std::cout << "P1 : " << p_matrixs[1] << std::endl;
  //std::cout << "0 Error : " << (SkewMatrix(obs[0]) * p_matrixs[0] * X).squaredNorm() << std::endl; 
  //std::cout << "1 Error : " << (SkewMatrix(obs[1]) * p_matrixs[1] * X).squaredNorm() << std::endl; 
  //std::cout << "P1 * X : " << p_matrixs[1] * X << std::endl;
  //std::cout << "Observation0 : " << obs[0] << std::endl;
  //std::cout << "Observation1 : " << obs[1] << std::endl;
  //std::cout << SkewMatrix(obs[1]) << std::endl;
  //std::cout << "SkewMatrix * P1 * X" << SkewMatrix(obs[1]) * p_matrixs[1] * X << std::endl;
  //std::cout << "X : " << X << std::endl;
  //std::cout << "DLT Formula Condition Number : " << singular_value(0) / singular_value(3) << std::endl;
  //std::cout << "DLT Geometry Error : " << TriangularGeometryError(p_matrixs, obs, X) << std::endl;
  return TriangularGeometryError(p_matrixs, obs, X);
}

void DLTTriangular(Eigen::Vector3d& lhs_obs, Eigen::Vector3d& rhs_obs, Mat34& lhs_p, Mat34& rhs_p, Eigen::Vector4d& X) {
  Eigen::Matrix<double, 4, 4> A;
  A.row(0) = lhs_obs(0) * lhs_p.row(2) - lhs_p.row(1);
  A.row(1) = lhs_p.row(0) - lhs_obs(1) * lhs_p.row(1);
  A.row(2) = rhs_obs(0) * rhs_p.row(2) - rhs_p.row(1);
  A.row(3) = rhs_p.row(0) - rhs_obs(1) * rhs_p.row(1);

  Eigen::JacobiSVD svd(A, Eigen::ComputeFullV);
  X = svd.matrixV().col(3);
  X = X / X(3);
}

void BundleAdjustmentTriangular(const std::vector<Mat34>& p_matrixs, const std::vector<Eigen::Vector3d>& obs, Eigen::Vector4d& X) {
    assert(p_matrixs.size() == obs.size());
    //DLT(p_matrixs, obs, X);
    //double x[3] = {X(0) / X(3), X(1) / X(3), X(2) / X(3)};
    double x[3] = {1.0, 1.0, 1.0};

    ceres::Problem problem;
    for (int i = 0; i < p_matrixs.size(); i++) {
      ceres::CostFunction* cost_fun =
          new ceres::AutoDiffCostFunction<ConstCameraMatrixCostFunctor, 2, 3>(
              new ConstCameraMatrixCostFunctor(p_matrixs[i], obs[i]));
      problem.AddResidualBlock(cost_fun, nullptr, x);
    }
    ceres::Solver::Options problem_options;
    problem_options.max_num_iterations = 500;
    ceres::Solver::Summary summary;
    ceres::Solve(problem_options, &problem, &summary);
    //std::cout << "Bundle AdjustmentTriangluar Report : " << summary.BriefReport() << std::endl;
    X << x[0], x[1], x[2], 1.0;

    double cost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    
    //std::cout << "BundleAdjustmentTriangular Error : " << TriangularGeometryError(p_matrixs, obs, X) << std::endl;
}