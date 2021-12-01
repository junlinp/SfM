#include <fstream>
#include <iostream>
#include <queue>

#include "Eigen/Dense"
#include "internal/function_programming.hpp"
#include "projective_constructure.hpp"
#include "ransac.hpp"
#include "sfm_data.hpp"
#include "sfm_data_io.hpp"
#include "solver/algebra.hpp"
#include "solver/fundamental_solver.hpp"
#include "solver/self_calibration_solver.hpp"
#include "solver/triangular_solver.hpp"
#include "solver/trifocal_tensor_solver.hpp"

bool ComputeFundamentalMatrix(const std::vector<KeyPoint>& lhs_keypoint,
                              const std::vector<KeyPoint>& rhs_keypoint,
                              Eigen::Matrix3d* fundamental_matrix,
                              std::vector<size_t>* inlier_index_ptr) {
  assert(lhs_keypoint.size() == rhs_keypoint.size());
  EigenAlignedVector<typename EightPointFundamentalSolver::DataPointType> datas;
  for (int i = 0; i < lhs_keypoint.size(); i++) {
    Eigen::Vector2d lhs_temp(lhs_keypoint[i].x, lhs_keypoint[i].y);
    Eigen::Vector2d rhs_temp(rhs_keypoint[i].x, rhs_keypoint[i].y);
    datas.push_back({lhs_temp, rhs_temp});
  }
  EightPointFundamentalSolver ransac_solver;

  ransac_solver.Fit(datas, *fundamental_matrix);
  double threshold = 9.49;

  std::vector<size_t> inlier_indexs;
  for (int i = 0; i < lhs_keypoint.size(); i++) {
    Eigen::Vector2d lhs_temp(lhs_keypoint[i].x, lhs_keypoint[i].y);
    Eigen::Vector2d rhs_temp(rhs_keypoint[i].x, rhs_keypoint[i].y);
    double error =
        SampsonError::Error({lhs_temp, rhs_temp}, *fundamental_matrix);
    if (error < threshold) {
      inlier_indexs.push_back(i);
    }
  }
  std::printf("inlier : %lu\n", inlier_indexs.size());
  bool ans = inlier_indexs.size() > 30;
  if (inlier_index_ptr != nullptr) {
    *inlier_index_ptr = std::move(inlier_indexs);
  }
  return ans;
}

bool ComputeFundamentalMatrix(const Matches& match, Mat33* fundamental_matrix,
                              Matches* inlier_match) {
  EigenAlignedVector<typename EightPointFundamentalSolver::DataPointType> datas;
  for (auto&& m : match) {
    datas.push_back({m.lhs_observation, m.rhs_observation});
  }
  EightPointFundamentalSolver ransac_solver;
  ransac_solver.Fit(datas, *fundamental_matrix);
  double threshold = 9.49;
  for (auto&& m : match) {
    double error = SampsonError::Error({m.lhs_observation, m.rhs_observation},
                                       *fundamental_matrix);
    if (error < threshold) {
      inlier_match->push_back(m);
    }
  }
  return inlier_match->size() > 30;
}

// a. choosing the initialize pair
// b. initialze the premary structure from the initialize pair
//
// c. Find the next image to register
// d. Find the 2D-3D corresponse between the registering image and the structure
// has build.
// e. Estimating the pose of registering image.
// f. Triangularing new sparse point.
// g. if there are some images need to register, then goto c, exit
// otherwise.

// a. choosing the initialize
// we will using the match only
void FindBestInitialPair(const std::map<Pair, Matches>& matches,
                         Pair* initial_pair) {
  if (initial_pair == nullptr) {
    return;
  } else {
    int matches_size = 0;
    for (auto&& [pair, matches] : matches) {
      if (matches.size() > matches_size) {
        *initial_pair = pair;
        matches_size = matches.size();
      }
    }
  }
}

// b. initialze the premary structure from the initialize pair
ProjectiveStructure InitializeStructure(const Pair& initial_pair,
                                        SfMData* sfm_data) {
  auto&& matches = sfm_data->matches.at(initial_pair);

  // Compute Fundamental Matrix and the inliers.
  Matches inliers_matches;
  Mat33 fundamental_matrix;
  // TODO (junlinp@qq.com):
  // whether using reverse match or other filter method to deal with 
  // the case that some lhs feature match to the same rhs feature.
  //
  ComputeFundamentalMatrix(matches, &fundamental_matrix, &inliers_matches);
  std::cout << "Initial Fundamental : " << fundamental_matrix << std::endl;
  
  std::printf("%lu inliers matches\n", inliers_matches.size());
  // return Structure struct
  Mat34 P1, P2;
  // P1 = [I | 0]
  // P2 = [ [e12]_x * F_12 + e12 * a^T | sigma * e12]
  // we using a^T = (0, 0, 0) and sigma = 1.0
  P1 << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  Eigen::Vector3d e12 = NullSpace(fundamental_matrix.transpose().eval());

  P2 << SkewMatrix(e12) * fundamental_matrix, e12;

  ProjectiveStructure projective_structure(*sfm_data);

  projective_structure.InitializeStructure(initial_pair, inliers_matches, P1,
                                           P2);

  auto Matches_2_3 = inliers_matches;
  auto Matches_2_4 = sfm_data->matches.at({2, 4});
  std::set<IndexT> s;
  for (Match m : Matches_2_3) {
    s.insert(m.lhs_idx);
  }
  int count = 0;
  for (Match m : Matches_2_4) {
    if (s.find(m.lhs_idx) != s.end()) {
      count++;
    }
  }
  std::printf("Count %d\n", count);
  return projective_structure;
}

// c. Find the next image to register
bool ProjectiveReconstruction(ProjectiveStructure& structure) {
  IndexT image_id = -1;
  while ((image_id = structure.NextImage()) != -1) {
    Correspondence cor = structure.FindCorrespondence(image_id);
    std::printf("Next Image %lld with correspondence %lu\n", image_id, cor.size());
    if (cor.size() < 10) {
      structure.UnRegister(image_id);
    } else {
      std::printf("Register Imaage %lld\n", image_id);
      // TODO: 2D-3D DLT
      // DLT from Correspondence
      // 
      std::vector<Observation> observation_2d = cor.Getobservations();
      std::vector<Eigen::Vector3d> points = cor.GetPoints();

      Mat34 P;

      structure.Register(image_id, P, cor);

      // f. Triangularing new sparse point.
      structure.TriangularNewPoint(image_id);

      // g. Refine Local Structure

      structure.LocalBundleAdjustment();
    }
  }
  return false;
}

// Union Find Set to construct track
// Define Hash and UnHash Function

//
/*
void BuildTrack(std::map<IndexT, std::vector<KeyPoint>> key_points,
                std::map<Pair, Matches> matches) {
  UnionFindSet<int64_t> ufs;

  for (auto pair : key_points) {
    int32_t image_idx = pair.first;
    const std::vector<KeyPoint>& key_point = pair.second;
    for (int32_t i = 0; i < key_point.size(); i++) {
      int64_t idx = Hash(image_idx, i);
      ufs.insertIdx(idx);
    }
  }

  // Union Find the matches
  for (auto match_pair : matches) {
    int32_t lhs_image_idx = match_pair.first.first;
    int32_t rhs_image_idx = match_pair.first.second;

    Matches match = match_pair.second;
    for (auto m : match) {
      int64_t lhs_hash_value = Hash(lhs_image_idx, m.first);
      int64_t rhs_hash_value = Hash(rhs_image_idx, m.second);
      ufs.Union(lhs_hash_value, rhs_hash_value);
    }
  }

  // show how many different set
  std::cout << ufs.DifferentSetSize() << std::endl;
}
*/

Eigen::Matrix3d VectorToSkewMatrix(const Eigen::Vector3d& v) {
  Eigen::Matrix3d res;
  double a1 = v(0), a2 = v(1), a3 = v(2);
  res << 0.0, -a3, a2, a3, 0.0, -a1, -a2, a1, 0.0;
  return res;
}

bool ComputeProjectiveReconstruction(const Eigen::Matrix3d& F, Mat34& P1,
                                     Mat34& P2) {
  // A. Compute Null Space of the traspose of F

  // B.Construct the {P1, P2} as follow:
  // P1 = [I | 0]
  // P2 = [(e')_x * F | e']
  Eigen::Vector3d e_dot = NullSpace(F);

  P1 << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  P2 << VectorToSkewMatrix(e_dot) * F, e_dot;
  return true;
}

bool ExistOneKey(const std::set<IndexT>& set, std::pair<IndexT, IndexT> item) {
  int lhs_exist = (set.find(item.first) != set.end());
  int rhs_exist = (set.find(item.second) != set.end());
  return lhs_exist ^ rhs_exist;
}

void Solve(const Mat34& A, const Eigen::Vector3d& b, Eigen::Vector4d& x) {
  Eigen::Matrix3d A33 = A.block(0, 0, 3, 3);

  Eigen::Vector3d x_ = A33.inverse() * (b - A.col(3));
  x << x_, 1.0;
}

void ComputeHTransform(const Mat34& sour, const Mat34& dest,
                       Eigen::Matrix4d& H) {
  // sour * H = dest
  // We assume H :
  //   h11  h12  h13  h14
  //   h21  h22  h23  h24
  //   h31  h32  h33  h34
  //   1.0  1.0  1.0  1.0

  Eigen::Vector4d h1, h2, h3, h4;
  Solve(sour, dest.col(0), h1);
  Solve(sour, dest.col(1), h2);
  Solve(sour, dest.col(2), h3);
  Solve(sour, dest.col(3), h4);
  H << h1, h2, h3, h4;
}

struct TripleIndex {
  IndexT I_, J_, K_;
  TripleIndex(IndexT I, IndexT J, IndexT K) {
    I_ = std::min(I, J);
    J_ = std::max(I, J);
    if (K < I_) {
      std::swap(I_, K);
    }

    if (K < J_) {
      std::swap(J_, K);
    }
    K_ = K;
  }
};
template <class T>
auto ReverseMatches(const std::vector<T>& input) {
  std::vector<T> output;
  output.reserve(input.size());
  for (T item : input) {
    std::swap(item.first, item.second);
    output.push_back(item);
  }
  return output;
}

template <typename Point>
void ToPly(const std::vector<Point>& points, const std::string& output) {
  std::ofstream ofs(output);
  // header
  ofs << "ply" << std::endl;
  ofs << "format ascii 1.0" << std::endl;
  ofs << "element vertex " << points.size() << std::endl;
  ofs << "property float x" << std::endl;
  ofs << "property float y" << std::endl;
  ofs << "property float z" << std::endl;
  ofs << "end_header" << std::endl;

  // body
  for (const Point& p : points) {
    ofs << p.x << " " << p.y << " " << p.z << std::endl;
  }

  ofs.close();
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
    return 1;
  }


  // TODO (junlinp@qq.com):
  // print out the scene graph status whether the matches
  // is connected.

  Pair initial_pair{0, 0};
  FindBestInitialPair(sfm_data.matches, &initial_pair);
  if (initial_pair.first == 0 && initial_pair.second == 0) {
    std::printf("We can't find a initial pair from the matches\n");
    return 1;
  } else {
    std::printf("Initial Structure with %lld - %lld\n", initial_pair.first,
                initial_pair.second);
  }
  ProjectiveStructure projective_structure =
      InitializeStructure(initial_pair, &sfm_data);
  bool b_construction = ProjectiveReconstruction(projective_structure);

  if (!b_construction) {
    std::printf("Projective Reconstruction Fails\n");
    return 1;
  }
  return 0;

  // Compute camera matrix K with the
  // IAC
  /*
  std::vector<Mat34> PS =
      projective_reconstruction |
      Transform([](const auto& iter) { return iter.second; }) | ToVector();

  Eigen::Matrix4d Q = IAC(PS, 3072, 2304);

  for (Mat34 P : PS) {
    std::cout << "P : " << P << std::endl;
    Eigen::Matrix3d omega = P * Q * P.transpose();
    std::cout << "omega : " << omega << std::endl;
    Eigen::Matrix3d K = RecoveryK(omega, 3072, 2304);
    std::cout << "K : " << K << std::endl;
  }
  */
  /*
  for (SparsePoint& point : sfm_data.structure_points) {
    Eigen::Vector3d p(point.x, point.y, point.z);
    Eigen::Vector4d new_p = H.inverse() * p.homogeneous();
    p = new_p.hnormalized();
    point.x = p(0);
    point.y = p(1);
    point.z = p(2);
  }
  */

  // triangulation
  //
  /*
  for (auto item_pair : fundamental_matrix) {
    auto match_pair = item_pair.first;
    auto matches = sfm_data.matches.at(match_pair);
    auto lhs_keypoint = sfm_data.key_points.at(match_pair.first);
    auto rhs_keypoint = sfm_data.key_points.at(match_pair.second);

    EigenAlignedVector<Mat34> p_matrixs;
    p_matrixs.push_back(projective_reconstruction.at(match_pair.first));
    p_matrixs.push_back(projective_reconstruction.at(match_pair.second));
    for (auto match : matches) {
      EigenAlignedVector<Eigen::Vector3d> obs;
      Eigen::Vector3d lhs_ob, rhs_ob;

      KeyPoint lhs = lhs_keypoint[match.first];
      KeyPoint rhs = rhs_keypoint[match.second];
      lhs_ob << lhs.x, lhs.y, 1.0;
      rhs_ob << rhs.x, rhs.y, 1.0;
      obs.push_back(lhs_ob);
      obs.push_back(rhs_ob);
      Eigen::Vector4d X;
      DLT(p_matrixs, obs, X);
      X.hnormalized();

      sfm_data.structure_points.emplace_back(X(0), X(1), X(2));
    }
  }
  */

  ToPly(sfm_data.structure_points, "./sparse_point.ply");

  // bundle adjustment the parameters rotation and translation

  Save(sfm_data, argv[1]);

  return 0;
}