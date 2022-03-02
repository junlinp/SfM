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
  static Eigen::Vector4d Evaluate(const TripleMatch& data_point, const Trifocal& model);
  static Eigen::Matrix<double, 4, 6> Jacobian(const TripleMatch& data_point, const Trifocal& model);
};

using TrifocalSampsonError = SampsonBase<TrifocalSamponErrorEvaluator,
                              TrifocalSamponErrorEvaluator>;
struct TrifocalError {

  // Compute the error of a data point according to the model
  static double Error(TripleMatch data_point, Trifocal model) { return TrifocalSampsonError::Error(data_point, model);}

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

class LinearSolver : public Typedefine {
 public:
  //static const size_t MINIMUM_DATA_POINT = 7;
  //using DataPointType = TripleMatch;
  //using ModelType = Trifocal;
  static void Fit(const std::vector<DataPointType>& data_points, ModelType* model);
};

class AlgebraMinimumSolver : public Typedefine {
public:
  static void Fit(const std::vector<DataPointType>& data_points, ModelType* model);
};

class BundleRefineSolver {
 public:
 static const size_t MINIMUM_DATA_POINT = 7;
 static const size_t MODEL_FREEDOM = 7;
 static const size_t MODEL_NUMBER = 18;
  using DataPointType = TripleMatch;
  using ModelType = Trifocal;
  static void Fit(const std::vector<DataPointType>& data_points, ModelType* model);
};

// TODO:
// Normalized Solver

using RansacTrifocalSolver = Ransac<BundleRefineSolver, TrifocalError>;

#endif  // SRC_SOLVER_TRIFOCAL_TENSOR_SOLVER_HPP_
