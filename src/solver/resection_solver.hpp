
#include <vector>

#include "Eigen/Dense"
#include "eigen_alias_types.hpp"

struct DLTSolver {
  using DataPointType = std::pair<Observation, Eigen::Vector3d>;
  static const size_t MINIMUM_DATA_POINT = 6;
  static const size_t MODEL_NUMBER = 1;
  static const size_t MODEL_FREEDOM = 2;
  using MODEL_TYPE = Mat34;
  using ModelType = MODEL_TYPE;
  static bool Fit(const std::vector<DataPointType>& data_points, MODEL_TYPE* model);
};

struct ReProjectiveError {
  using DataPointType = std::pair<Observation, Eigen::Vector3d>;
  static double Error(const std::pair<Observation, Eigen::Vector3d>& data_point,
               const Mat34& P);

   // we assume that the point will lay in 1px with probability 99.7%
   static bool RejectRegion(double error) {
     return error * 0.5 * 0.5 > 5.99;
   }

};

struct RansacResection {
  //bool Resection(const std::vector<Observation>& observations,
  //               const std::vector<Eigen::Vector3d>& points, Mat34& P);
  bool Resection(const std::vector<std::pair<Observation, Eigen::Vector3d>>& data_points, Mat34& P, std::vector<size_t>* inliers_index);
};