#ifndef SRC_SOLVER_TRIFOCAL_TENSOR_SOLVER_HPP_
#define SRC_SOLVER_TRIFOCAL_TENSOR_SOLVER_HPP_

#include <vector>
#include "trifocal_tensor.hpp"
#include "ransac.hpp"
#include "algebra.hpp"



std::ostream& operator<<(std::ostream& os, Trifocal tirfocal);

double GeometryError(const TripleMatch data_point, Trifocal& model);

void RecoveryCameraMatrix(const Trifocal& trifocal, Mat34& P1, Mat34& P2,
                          Mat34& P3);

void BundleRecovertyCameraMatrix(const std::vector<TripleMatch>& data_points,
                                 const Trifocal& trifocal, const Mat34& P1,
                                 const Mat34& P2, Mat34& P3);

struct TrifocalSamponErrorEvaluator {
  static Eigen::VectorXd Evaluate(const TripleMatch& data_point, const Trifocal& model);
  static Eigen::Matrix<double, 9, 6> Jacobian(const TripleMatch& data_point, const Trifocal& model);
};

using TrifocalSampsonError = SampsonBase<TrifocalSamponErrorEvaluator,
                              TrifocalSamponErrorEvaluator>;

struct TrifocalReprojectiveErrorEvaluator {
  static double Error(TripleMatch data_point, Trifocal model);
};

struct TrifocalError {

  // Compute the error of a data point according to the model
  static double Error(TripleMatch data_point, Trifocal model) {
    //return TrifocalSampsonError::Error(data_point, model);
    return TrifocalReprojectiveErrorEvaluator::Error(data_point, model);
  }

  // Whether this error will be reject
  static bool RejectRegion(double error) {
    // sigma is 0.5
    return error > 12.59 * 0.5 * 0.5;
  }

};

class Typedefine {
public:
  static const size_t MINIMUM_DATA_POINT = 7;
  using DataPointType = TripleMatch;
  using ModelType = Trifocal;
};

template<typename T>
inline std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> ConvertToMatrix(const std::vector<T> data_points) {
  size_t n = data_points.size();
  Eigen::MatrixXd lhs(2, n), middle(2, n), rhs(2, n);

  size_t index = 0;
  for (T point : data_points) {
    lhs.col(index) = point.I_observation;
    middle.col(index) = point.J_observation;
    rhs.col(index) = point.K_observation;
    index++;
  }

  return {lhs, middle, rhs};
}

inline std::vector<TripleMatch> ConvertToDataPoint(Eigen::MatrixXd lhs, Eigen::MatrixXd middle, Eigen::MatrixXd rhs) {
  std::vector<TripleMatch> res;

  size_t n = lhs.cols();
  res.reserve(n);
  for (size_t i = 0; i < n; i++) {
    TripleMatch t;
    t.I_observation = lhs.col(i).hnormalized();
    t.J_observation = middle.col(i).hnormalized();
    t.K_observation = rhs.col(i).hnormalized();
    res.push_back(t);
  }
  return res;
}

template<typename Solver>
class TrifocalNormalized : public Typedefine {
  public:
  static void Fit(const std::vector<DataPointType>& data_points, ModelType* model) {

  Eigen::Matrix3d H, H_dot, H_dot_dot;
  
  auto&& [lhs_point, middle_point, rhs_point] = ConvertToMatrix(data_points);

  H = NormalizedCenter(lhs_point);
  H_dot = NormalizedCenter(middle_point);
  H_dot_dot = NormalizedCenter(rhs_point);

  Eigen::MatrixXd lhs_t(3, lhs_point.cols());
  lhs_t = lhs_point.colwise().homogeneous();
  Eigen::MatrixXd middle_t(3, middle_point.cols());
  middle_t = middle_point.colwise().homogeneous();
  Eigen::MatrixXd rhs_t(3, rhs_point.cols()) ;
  rhs_t = rhs_point.colwise().homogeneous();

  auto normalized_data_points = ConvertToDataPoint(H * (lhs_t), H_dot * (middle_t), H_dot_dot * (rhs_t));
  Solver::Fit(normalized_data_points, model);
  Eigen::Matrix3d H_dot_inverse = H_dot.inverse(), H_dot_dot_inverse = H_dot_dot.inverse();
  Eigen::Matrix3d T[3];
  Eigen::Matrix3d T_hat[3] = {model->lhs, model->middle, model->rhs};
  for (int i = 0; i < 3; i++) for(int j = 0; j < 3; j++ ) for (int k = 0; k < 3; k++) {
    T[i](j, k) = 0.0;
    for (int r = 0; r < 3; r++) for (int s = 0; s < 3; s++) for (int t = 0 ; t < 3; t++) { 
      T[i](j, k) += H(r, i) * H_dot_inverse(j, s) * H_dot_dot_inverse(k, t) * T_hat[r](s, t);
    }
  }

  model->lhs = T[0];
  model->middle = T[1];
  model->rhs = T[2];
  }
};

class LinearSolver : public Typedefine {
 public:
  //static const size_t MINIMUM_DATA_POINT = 7;
  //using DataPointType = TripleMatch;
  //using ModelType = Trifocal;
  static void Fit(const std::vector<DataPointType>& data_points, ModelType* model);
};
using NormalizedLinearSolver = TrifocalNormalized<LinearSolver>;

class AlgebraMinimumSolver : public Typedefine {
public:
  static void Fit(const std::vector<DataPointType>& data_points, ModelType* model);
};

class BundleRefineSolver {
 public:
 static const size_t MINIMUM_DATA_POINT = 7;
 static const size_t MODEL_FREEDOM = 7;
 static const size_t MODEL_NUMBER = 1;
  using DataPointType = TripleMatch;
  using ModelType = Trifocal;
  static void Fit(const std::vector<DataPointType>& data_points, ModelType* model);
};
// TODO:
// Normalized Solver

using RansacTrifocalSolver = Ransac<BundleRefineSolver, TrifocalError>;

#endif  // SRC_SOLVER_TRIFOCAL_TENSOR_SOLVER_HPP_
