#include "triangular_solver.hpp"
#include "ceres/problem.h"
#include "ceres/autodiff_cost_function.h"
#include "ceres/solver.h"
#include <assert.h>
void DLT(const EigenAlignedVector<Mat34>& p_matrixs,
         const EigenAlignedVector<Eigen::Vector3d>& obs, Eigen::Vector4d& X) {
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
    coefficent.row(2 * i + 0) = u * P.row(2) - P.row(1);
    coefficent.row(2 * i + 1) = P.row(0) - v * P.row(1);
  }
  // std::cout << "DLT coefficnet : " << coefficent << std::endl;
  Eigen::JacobiSVD svd(coefficent, Eigen::ComputeFullV);
  //std::cout << "Singular Value : " << svd.singularValues() << std::endl;
  //std::cout << "Zero Value : " << svd.matrixV().row(3) << std::endl;

  //X = coefficent.colPivHouseholderQr().solve(zeros_constant);
  X = svd.matrixV().row(3);
  // std::cout << "X : " << X << std::endl;
  X(0) /= X(3);
  X(1) /= X(3);
  X(2) /= X(3);
  X(3) = 1.0;
}
namespace {
    struct BundleAdjustmentTriangularCost {
        BundleAdjustmentTriangularCost(const Mat34& p_matrixs,const Eigen::Vector3d& obs) : p_matrixs_(p_matrixs), obs_(obs) {}

        Mat34 p_matrixs_;
        Eigen::Vector3d obs_;


        // X is 3-dimensional vector
        template<typename T>
        bool operator()(const T* X, T* output) const {

            T x_[4] = {X[0], X[1], X[2], T(1.0)};

            T product[3];

            for (int i = 0; i < 3; i++) {
                product[i] = T(0.0);
                for (int k = 0; k < 4; k++) {
                    product[i] += T(p_matrixs_(i, k)) * x_[k];
                }
            }


            product[0] /= product[2];
            product[1] /= product[2];

            output[0] = product[0] - T(obs_(0));
            output[1] = product[1] - T(obs_(1));
            
            return true;
        }
    };
}
void BundleAdjustmentTriangular(const EigenAlignedVector<Mat34>& p_matrixs, const EigenAlignedVector<Eigen::Vector3d>& obs, Eigen::Vector4d& X) {
    assert(p_matrixs.size() == obs.size());
    DLT(p_matrixs, obs, X);
    double x[3] = {X(0) / X(3), X(1) / X(3), X(2) / X(3)};

    ceres::Problem problem;
    for (int i = 0; i < p_matrixs.size(); i++) {
      ceres::CostFunction* cost_fun =
          new ceres::AutoDiffCostFunction<BundleAdjustmentTriangularCost, 2, 3>(
              new BundleAdjustmentTriangularCost(p_matrixs[i], obs[i]));
      problem.AddResidualBlock(cost_fun, nullptr, x);
    }
    ceres::Solver::Options problem_options;
    ceres::Solver::Summary summary;
    ceres::Solve(problem_options, &problem, &summary);

    X << x[0], x[1], x[2], 1.0;
}