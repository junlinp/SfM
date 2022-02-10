#ifndef SRC_SOLVER_TRIFOCAL_TENSOR_SOLVER_HPP_
#define SRC_SOLVER_TRIFOCAL_TENSOR_SOLVER_HPP_

#include <vector>

#include "trifocal_tensor.hpp"




std::ostream& operator<<(std::ostream& os, Trifocal tirfocal);

double Error(TripleMatch data_point, Trifocal model);

double GeometryError(const TripleMatch data_point, Trifocal& model);

void RecoveryCameraMatrix(const Trifocal& trifocal, Mat34& P1, Mat34& P2,
                          Mat34& P3);

void BundleRecovertyCameraMatrix(const std::vector<TripleMatch>& data_points,
                                 const Trifocal& trifocal, const Mat34& P1,
                                 const Mat34& P2, Mat34& P3);
class LinearSolver {
 public:
  using DataPointType = TripleMatch;
  using ModelType = Trifocal;
  void Fit(const std::vector<DataPointType>& data_points, ModelType& model);
};

class BundleRefineSolver {
 public:
  using DataPointType = TripleMatch;
  using ModelType = Trifocal;
  void Fit(const std::vector<DataPointType>& data_points, ModelType& model);
};
#endif  // SRC_SOLVER_TRIFOCAL_TENSOR_SOLVER_HPP_
