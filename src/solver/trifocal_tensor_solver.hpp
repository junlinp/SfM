#ifndef SRC_SOLVER_TRIFOCAL_TENSOR_SOLVER_HPP_
#define SRC_SOLVER_TRIFOCAL_TENSOR_SOLVER_HPP_

#include <vector>
#include "trifocal_tensor.hpp"
#include "ransac.hpp"




std::ostream& operator<<(std::ostream& os, Trifocal tirfocal);

double GeometryError(const TripleMatch data_point, Trifocal& model);

void RecoveryCameraMatrix(const Trifocal& trifocal, Mat34& P1, Mat34& P2,
                          Mat34& P3);

void BundleRecovertyCameraMatrix(const std::vector<TripleMatch>& data_points,
                                 const Trifocal& trifocal, const Mat34& P1,
                                 const Mat34& P2, Mat34& P3);

struct TrifocalError {

  // Compute the error of a data point according to the model
  static double Error(TripleMatch data_point, Trifocal model);

  // Whether this error will be reject
  static bool RejectRegion(double error);
};

class LinearSolver {
 public:
 static const size_t MINIMUM_DATA_POINT = 7;
  using DataPointType = TripleMatch;
  using ModelType = Trifocal;
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
