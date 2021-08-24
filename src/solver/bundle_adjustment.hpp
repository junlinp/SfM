#ifndef SRC_SOLVER_BUNDLE_ADJUSTMENT_HPP_
#define SRC_SOLVER_BUNDLE_ADJUSTMENT_HPP_

#include <vector>
#include <set>
#include "eigen_alias_types.hpp"
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
        ConstCameraMatrixCostFunctor(const Mat34& p_matrixs,const Eigen::Vector3d& obs) : p_matrixs_(p_matrixs), obs_(obs) {}
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
struct CostFunctor {
  Eigen::Vector3d obs_;
  CostFunctor(const Eigen::Vector3d& obs_) : obs_(obs_) {}

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

#endif  // SRC_SOLVER_BUNDLE_ADJUSTMENT_HPP_
