#ifndef SRC_SOLVER_BUNDLE_ADJUSTMENT_HPP_
#define SRC_SOLVER_BUNDLE_ADJUSTMENT_HPP_

#include <vector>
#include <set>

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
#endif  // SRC_SOLVER_BUNDLE_ADJUSTMENT_HPP_
