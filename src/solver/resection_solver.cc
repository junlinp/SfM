#include "resection_solver.hpp"

#include "algebra.hpp"
#include "internal/function_programming.hpp"
#include "ransac.hpp"

void DLT(std::vector<Eigen::Vector2d> observations,
         std::vector<Eigen::Vector3d> points,
         Eigen::Matrix<double, 3, 4, Eigen::RowMajor>& P) {
  assert(observations.size() == points.size());
  int n = observations.size();
  Eigen::MatrixXd A(3 * n, 12);
  A.setZero();
  for (size_t i = 0; i < n; i++) {
    const Eigen::Vector2d& ob = observations[i];
    double u = ob.x();
    double v = ob.y();

    Eigen::Vector4d X = points[i].homogeneous();

    A.block(3 * i + 0, 4, 1, 4) = -X.transpose().eval();
    A.block(3 * i + 0, 8, 1, 4) = u * X.transpose().eval();
    A.block(3 * i + 1, 0, 1, 4) = X.transpose().eval();
    A.block(3 * i + 1, 8, 1, 4) = -v * X.transpose().eval();
    A.block(3 * i + 2, 0, 1, 4) = -u * X.transpose().eval();
    A.block(3 * i + 2, 4, 1, 4) = v * X.transpose().eval();
  }

  Eigen::VectorXd p_coeffient = NullSpace(A);

  P = Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(
      p_coeffient.data());
}

bool DLTSolver::Fit(const std::vector<DataPointType>& data_points,
                    MODEL_TYPE* model) {
  if (model == nullptr) {
    return false;
  }
  std::vector<Observation> obs =
      data_points | Transform([](auto&& item) { return item.first; }) |
      ToVector();
  std::vector<Eigen::Vector3d> points =
      data_points | Transform([](auto&& item) { return item.second; }) |
      ToVector();
  DLT(obs, points, *model);
  return true;
}
namespace resection {

double ReProjectiveError::Error(const DataPointType& data_point,
                                const Mat34& P) {
  Eigen::Vector3d uv = P * data_point.second.homogeneous();
  return (data_point.first - uv.hnormalized()).squaredNorm();
}
}

//bool RansacResection::Resection(
//    const std::vector<Observation>& observations,
//    const std::vector<Eigen::Vector3d>& points, Mat34& P) {
//  std::vector<size_t> placeholder;
//  using DATA_TYPE = std::pair<Observation, Eigen::Vector3d>;
//  std::vector<DATA_TYPE> data_points;
//  for (size_t i = 0; i < observations.size(); i++) {
//    data_points.push_back({observations[i], points[i]});
//  }
//  return Ransac<DLTSolver, ReProjectiveError>::Inference(data_points, placeholder, &P);
//}

bool RansacResection::Resection(const std::vector<std::pair<Observation, Eigen::Vector3d>>& data_points, Mat34& P, std::vector<size_t>* inliers_index) {
  std::vector<size_t> placeholder;
  return Ransac<DLTSolver, resection::ReProjectiveError>::Inference(data_points, inliers_index == nullptr ? placeholder : *inliers_index, &P);
}