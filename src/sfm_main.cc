#include <iostream>

#include "sfm_data.hpp"
#include "sfm_data_io.hpp"

#include "Eigen/Dense"
#include "solver/fundamental_solver.hpp"
#include "ransac.hpp"

bool ComputeFundamentalMatrix(const std::vector<KeyPoint>& lhs_keypoint,
                              const std::vector<KeyPoint>& rhs_keypoint,
                              Eigen::Matrix3d* fundamental_matrix) {
  assert(lhs_keypoint.size() == rhs_keypoint.size());

  std::vector<typename EightPointFundamentalSolver::DataPointType>  datas;
  for(int i = 0; i < lhs_keypoint.size(); i++) {
    datas.push_back({lhs_keypoint[i], rhs_keypoint[i]});
  }
  Ransac<EightPointFundamentalSolver> ransac_solver;
  std::vector<size_t> inlier_indexs;
  ransac_solver.Inference(datas, inlier_indexs, fundamental_matrix);

  return inlier_indexs.size() > 30;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    return 1;
  }

  SfMData sfm_data;
  bool b_load = Load(sfm_data, argv[1]);
  if (b_load) {
    std::printf("Load Sfm Data Finish\n");
  } else {
    std::printf("Load Sfm Data From %s Fails\n", argv[1]);
  }
    // Filter With fundamental matrix
    std::map<Pair, Eigen::Matrix3d> fundamental_matrix;
    for (const auto& iter : sfm_data.matches) {
        Pair pair = iter.first;
        if (iter.second.size() < 8) {
            continue;
        }
        std::printf("Compute Fundamental Matrix for pair <%lld, %lld>\n", pair.first, pair.second);
        const std::vector<KeyPoint>& lhs_keypoints = sfm_data.key_points.at(pair.first);
        const std::vector<KeyPoint>& rhs_keypoints = sfm_data.key_points.at(pair.second);

        std::vector<KeyPoint> lhs, rhs;
        lhs.reserve(iter.second.size());
        rhs.reserve(iter.second.size());
        for (const Matche& m : iter.second) {
            lhs.push_back(lhs_keypoints[m.first]);
            rhs.push_back(rhs_keypoints[m.second]);
        }

        Eigen::Matrix3d F;
        if (ComputeFundamentalMatrix(lhs, rhs, &F)) {
            fundamental_matrix.insert({pair, F});
        }
    }
    std::printf("%lu Matches are reserved after fundamental matrix filter\n", fundamental_matrix.size());

    // whether the graph is connected.

    // Compute the P matrix for each images
    // using fundamental matrix.

    // Compute camera matrix K with the 
    // IAC

    // bundle adjustment the parameters rotation and translation
    Save(sfm_data, argv[1]);

    return 0;
}