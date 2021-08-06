#include <iostream>

#include "sfm_data.hpp"
#include "sfm_data_io.hpp"

#include "Eigen/Dense"
#include "solver/fundamental_solver.hpp"
#include "ransac.hpp"
#include "internal/function_programming.hpp"

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
  std::printf("inlier : %lu\n", inlier_indexs.size());
  return inlier_indexs.size() > 30;
}
using Mat34 = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;

Eigen::Matrix3d VectorToSkewMatrix(const Eigen::Vector3d& v) {
  Eigen::Matrix3d res;
  double a1 = v(0), a2 =  v(1), a3 = v(2);
  res << 0.0, -a3, a2,
         a3 ,  0.0, -a1,
         -a2, a1, 0.0;
  return res;
}

Eigen::Vector3d NullSpace(const Eigen::Matrix3d& m) {
  Eigen::JacobiSVD svd(m, Eigen::ComputeFullV);
  return svd.matrixV().col(2);
}

bool ComputeProjectiveReconstruction(const Eigen::Matrix3d& F, Mat34& P1, Mat34& P2) {
  // A. Compute Null Space of the traspose of F

  // B.Construct the {P1, P2} as follow:
  // P1 = [I | 0]
  // P2 = [(e')_x * F | e']
  Eigen::Vector3d e_dot = NullSpace(F);

  P1 << 1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0;
  P2 << VectorToSkewMatrix(e_dot) * F, e_dot;
  return true;
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
        if (iter.second.size() < 30) {
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

    std::set<IndexT> id_set;
    auto v = fundamental_matrix | Transform([](auto item){ return item.first;}) | ToVector();
    for (auto i : v) {
      id_set.insert(i.first);
      id_set.insert(i.second);
    }
    // whether the graph is connected.
    if (id_set.size() != sfm_data.views.size()) {
      std::printf("Warning: The visible graph is not connected after filter\n");
    }

    // Compute the P matrix for each images
    // using fundamental matrix.
    std::map<IndexT, Mat34> project_reconstruction;

    // Compute camera matrix K with the 
    // IAC

    // bundle adjustment the parameters rotation and translation
    Save(sfm_data, argv[1]);

    return 0;
}