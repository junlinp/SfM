#ifndef SRC_SOLVER_BUNDLE_ADJUSTMENT_HPP_
#define SRC_SOLVER_BUNDLE_ADJUSTMENT_HPP_

#include <vector>
#include <set>
#include "eigen_alias_types.hpp"
#include "ceres/rotation.h"

/*
struct ParameterBlock {
  double* ptr_;
  size_t size_;
  ParameterBlock(double* ptr, size_t size) : ptr_(ptr), size_(size) {}
};

template<class Functor>
struct ResidualBlock {
    Functor functor_;
    std::vector<double*> parameters_;

    ResidualBlock(Functor functor, double* x0) : functor_(functor), parameters_{x0} {};
    ResidualBlock(Functor functor, double* x0, double* x1)
        : functor_(functor), parameters_{x0, x1} {};
};

class CostFunction {
 public:
  void operator()(double* x0, double* residual);
  void operator()(double* x0, double* x1, double* residual);
  void operator()(double* x0, double* x1, double* x2, double* residual);
  void operator()(double* x0, double* x1, double* x2, double* x3,
                  double* residual);
  void operator()(double* x0, double* x1, double* x2, double* x3, double* x4,
                  double* residual);
  void operator()(double* x0, double* x1, double* x2, double* x3, double* x4,
                  double* x5, double* residual);
};

template<class Functor, int residual_num, int x0_num, int x1_num>
class AutoDiffCostFunction : public : CostFunction {
    
};

class Problem {
public:

    void AddParameterBlock(double* ptr, size_t size) {
        if (parameter_filter_.find(ptr) == parameter_filter_.end()) {
          parameter_block_.emplace_back(ptr, size);
        }
    }

    template<class Functor, class... Args>
    void AddResidualBlock(Functor&& functor, Args... args) {
        ResidualBlock<Functor> block(std::forward<Functor>(functor), args...);

        residual_blocks_.push_back(block);
    }

    void setParameterConstant(double* ptr, std::vector<size_t>& indexs);

    void Solve();

    std::set<double*> parameter_filter_;
    std::vector<ParameterBlock> parameter_block_;
};
*/


struct ConstCameraMatrixCostFunctor {
        explicit ConstCameraMatrixCostFunctor(const Mat34& p_matrixs,const Eigen::Vector3d& obs) : p_matrixs_(p_matrixs), obs_(obs) {}
        explicit ConstCameraMatrixCostFunctor(const Mat34& p_matrixs,const Eigen::Vector2d& obs) : p_matrixs_(p_matrixs), obs_(obs.homogeneous()) {}
        Mat34 p_matrixs_;
        Eigen::Vector3d obs_;
        // X is 3-dimensional vector
        template<typename T>
        bool operator()(const T* X, T* output) const {
            T product[3];
            for (int i = 0; i < 3; i++) {
                product[i] = T(p_matrixs_(i, 3));
                for (int k = 0; k < 3; k++) {
                    product[i] += T(p_matrixs_(i, k)) * X[k];
                }
            }

            output[0] = product[0] - T(obs_(0)) * product[2];
            output[1] = product[1] - T(obs_(1)) * product[2];
            return true;
        }
    };
struct CostFunctor {
  Eigen::Vector3d obs_;
  explicit CostFunctor(const Eigen::Vector3d& obs_) : obs_(obs_) {}
  explicit CostFunctor(const Eigen::Vector2d& obs_) : obs_(obs_.homogeneous()) {}

  template<typename T>
  bool operator()(const T* P, const T* X, T* output) const {
    T x_[4] = {X[0], X[1], X[2], T(1.0)};
    T product[3];
    for (int i = 0; i < 3; i++) {
      product[i] = T(0.0);
      for (int k = 0; k < 4; k++) {
        product[i] += P[i * 4 + k] * x_[k];
      }
    }

    product[0] /= product[2];
    product[1] /= product[2];
    output[0] = product[0] - T(obs_(0));
    output[1] = product[1] - T(obs_(1));
    return true;
  }
};


template<typename T>
bool ReProjectiveCostFunction(const T* fx,const T*fy,const T*cx,const T*cy,const T* angle_axis,const T* center,const T* X,const T* obs, T* res) {
  T x[3] = {X[0] - center[0], X[1] - center[1], X[2] - center[2]};
  T x_[3] = {T(0.0)};
  ceres::AngleAxisRotatePoint(angle_axis, x, x_);

  T u = fx[0] * x_[0] / x_[2] + cx[0];
  T v = fy[0] * x_[1] / x_[2] + cy[0];

  res[0] = u - obs[0];
  res[1] = v - obs[1];
  return true;
}


struct ConstRCCostFunctor {
  double angle_axis[3];
  double center[3];
  double obs[2];

  ConstRCCostFunctor(double* angle_axis, double* center, double* obs)
      : angle_axis{angle_axis[0], angle_axis[1], angle_axis[2]}, center{center[0], center[1], center[2]}, obs{obs[0], obs[1]} {};

  template<typename T>
  bool operator()(const T* K,const T* X, T* residual) const {
    const T T_angle_axis[3] = {T(angle_axis[0]),T(angle_axis[1]),T(angle_axis[2])};
    const T T_center[3] = {T(center[0]),T(center[1]),T(center[2])};
    const T T_obs[2] = {T(obs[0]),T(obs[1])};

    return ReProjectiveCostFunction(K, K + 1, K + 2, K + 3, T_angle_axis, T_center, X, T_obs, residual);
  }
};

struct FreeCostFunctor {
  double obs[2];
  FreeCostFunctor(double* obs) : obs{obs[0], obs[1]} {};
  template<typename T>
  bool operator()(const T* K, const T* angle_axis, const T* center, const T* X, T* residual) const {
    const T T_obs[2] = {T(obs[0]),T(obs[1])};
    return ReProjectiveCostFunction(K, K + 1, K + 2, K + 3, angle_axis, center, X, T_obs, residual);
  }
};



#endif  // SRC_SOLVER_BUNDLE_ADJUSTMENT_HPP_
