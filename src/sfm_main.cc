#include <fstream>
#include <iostream>

#include "Eigen/Dense"
#include "internal/function_programming.hpp"
#include "ransac.hpp"
#include "sfm_data.hpp"
#include "sfm_data_io.hpp"
#include "solver/algebra.hpp"
#include "solver/fundamental_solver.hpp"
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
  if (inlier_index_ptr) {
    *inlier_index_ptr = std::move(inlier_indexs);
  }
  return inlier_indexs.size() > 30;
}

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

#include <queue>

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
  }
};
void ProjectiveReconstruction(
    const std::map<Pair, Eigen::Matrix3d>& fundamental_matrix,
    const std::map<Piar, Matches>& filter_matches,
    const std::map<IndexT, KeyPoint>& keypoint,
    std::map<IndexT, Mat34>& projective_reconstruction) {
  using Fundamental_Matrix_Type =
      typename std::map<Pair, Eigen::Matrix3d>::value_type;

  // A. find all the triple using the fundamental_matrix.
  // B. find a best triple to initialize.
  // C. Init three camera using trifocal.
  // D. tranverse the rest of triple which two of that has been initilized.
  // E. go to D until all tirple is processed.

  // A. find all the triple using the fundamental_matrix
  std::set<IndexT> index_set;
  auto all_index = fundamental_matrix |
                   Transform([](auto& item) { return iter.first }) | ToVector();
  for (auto& item : all_index) {
    index_set.insert(item.first);
    index_set.insert(item.second);
  }

  std::vector<TripleIndex> triple_pair;
  for (IndexT I : index_set) {
    for (IndexT J : index_set) {
      if (I != J && fundamental_matrix.find({std::min(I, J), std::max(I, J)}) !=
                        fundamental_matrix.end()) {
        for (IndexT K : index_set) {
          if (I != K && J != K) {
            Pair one = {std::min(I, K), std::max(I, K)};
            Pair two = {std::min(J, K), std::max(J, K)};

            if (fundamental_matrix.find(one) != fundamental_matrix.end() &&
                fundamental_matrix.find(tow) != fundamental_matrix.end()) {
              TripleIndex triple = {I, J, K};
              triple_pair.push_back(triple);
            }
          }
        }
      }
    }
  }

  // B.find a best triple to initialize.
  std::set<IndexT> used_index;
  TripleIndex initial = triple_pair.front();
  used_index.insert(initial.I_);
  used_index.insert(initial.J_);
  used_index.insert(initial.K_);

  auto TrackBuilder = [](std::vector<KeyPoint>& I_keypoint,
                         std::vector<KeyPoint>& J_keypoint,
                         std::vector<KeyPoint>& K_keypoint,
                         Matches& I_to_J_matches, Matches& J_to_K_matches) {
    std::vector<TriPair> res;
    std::map<IndexT, IndexT> J_to_K_matches_map;
    for (Matche match : J_to_K_matches) {
      J_to_K_matches_map[match.first] = match.second;
    }

    for (Matche match : I_to_J_matches) {
      if (J_to_K_matches_map.find(match.second) != J_to_K_matches_map.end()) {
        IndexT k = J_to_K_matches_map[match.second];
        Eigen::Vector2d I_obs(I_keypoint[match.first].x,
                              I_keypoint[match.first].y);
        Eigen::Vector2d J_obs(J_keypoint[match.second].x,
                              J_keypoint[match.second].y);
        Eigen::Vector2d K_obs(K_keypoint[k].x, K_keypoint[k].y);
        TriPair p(I_obs, J_obs, K_obs);
        res.push_back(p);
      }
    }
    return res;
  };

  {
    std::vector<KeyPoint> I_keypoint = keypoint[initial.I_];
    std::vector<KeyPoint> J_keypoint = keypoint[initial.J_];
    std::vector<KeyPoint> K_keypoint = keypoint[initial.K_];

    std::vector<TriPair> data_points =
        TrackBuilder(I_keypoint, J_keypoint, K_keypoint,
                     filter_matches[{initial.I_, initial.J_}],
                     filter_matches[{initial.J_, initial.K_}]);
    Trifocal trifocal;
    BundleRefineSolver solver;
    solver.Fit(data_points, trifocal);
    RecoveryCameraMatrix(trifocal, projective_reconstruction[initial.I_],
                         projective_reconstruction[initial.J_],
                         projective_reconstruction[initial.K_]);
  }

  auto ContanerExist(auto container, auto item) {
    return container.find(item) != container.end();
  };
  // remove the initial pair
  triple_pair.erase(triple_pair.begin());
  while (true) {
    bool found = false;

    for (TripleIndex triple_index : triple_pair) {
      if (TripleValid(triple_index, used_index)) {
        IndexT first, second, need_to_predict;
        if (!ContanerExist(used_index, triple_index.I_)) {
          first = triple_index.J_;
          second = triple_index.K_;
          need_to_predict = triple_index.I_;
        }

        if (!ContanerExist(used_index, triple_index.J_)) {
          first = triple_index.I_;
          second = triple_index.K_;
          need_to_predict = triple_index.J_;
        }
        if (!ContanerExist(used_index, triple_index.K_)) {
          first = triple_index.I_;
          second = triple_index.J_;
          need_to_predict = triple_index.K_;
        }
        std::vector<KeyPoint> first_keypoint = keypoint[first];
        std::vector<KeyPoint> second_keypoint = keypoint[second];
        std::vector<KeyPoint> need_to_predict_keypoint = keypoint[need_to_predict];
        std::vector<TriPair> data_point = TrackBuilder(
            first_keypoint, second_keypoint, need_to_predict_keypoint,
            filter_matches[{first, second}],
            filter_matches[{second, need_to_predict}]);

        Trifocal trifocal;
        BundleRefineSolver solver;
        solver.Fit(data_point, trifocal);

        BundleRecovertyMatrix(data_point, trifocal, prejective_matrix[first], prejective_construction[second], p[need_to_predict]);

        // append
        found = true;
        used_index.insert(triple_index.I_);
        used_index.insert(triple_index.J_);
        used_index.insert(triple_index.K_);
      }
      // TODO: global bundle adjustment after adding a new camera.
    }
  }
  // Triple
  /*
  std::queue<Fundamental_Matrix_Type> que;
  for (auto item : fundamental_matrix) {
    que.push(item);
  }
  std::set<IndexT> processed_index;

  // A. Init
  Fundamental_Matrix_Type init_seed = que.front();
  que.pop();
  IndexT lhs_index = init_seed.first.first;
  IndexT rhs_index = init_seed.first.second;
  ComputeProjectiveReconstruction(init_seed.second,
                                  projective_reconstruction[lhs_index],
                                  projective_reconstruction[rhs_index]);
  processed_index.insert(lhs_index);
  processed_index.insert(rhs_index);
  //
  bool found_and_continue = true;

  while (found_and_continue) {
    found_and_continue = false;
    std::queue<Fundamental_Matrix_Type> temp_que;
    while (!que.empty()) {
      Fundamental_Matrix_Type need_to_process = que.front();
      que.pop();

      if (ExistOneKey(processed_index, need_to_process.first)) {
        found_and_continue = true;
        IndexT lhs_index = need_to_process.first.first;
        IndexT rhs_index = need_to_process.first.second;

        Mat34 P1, P2;
        ComputeProjectiveReconstruction(need_to_process.second, P1, P2);
        if (processed_index.find(lhs_index) == processed_index.end()) {
          Eigen::Matrix4d H;
          ComputeHTransform(P2, projective_reconstruction.at(rhs_index), H);
          projective_reconstruction[lhs_index] = P1 * H;

        } else {
          Eigen::Matrix4d H;
          ComputeHTransform(P1, projective_reconstruction.at(lhs_index), H);
          projective_reconstruction[rhs_index] = P2 * H;
        }

        processed_index.insert(lhs_index);
        processed_index.insert(rhs_index);
      } else {
        temp_que.push(need_to_process);
      }
    }
    que = std::move(temp_que);
  }
  */
}
// row and col start with 1 instead of 0
Eigen::Matrix<double, 1, 16> GenerateCoeffient(const Mat34 P, size_t row,
                                               size_t col) {
  Eigen::Matrix<double, 1, 16> res;
  Eigen::Vector4d rhs = P.row(col - 1);
  res << P(row - 1, 0) * rhs.transpose(), P(row - 1, 1) * rhs.transpose(),
      P(row - 1, 2) * rhs.transpose(), P(row - 1, 3) * rhs.transpose();
  return res;
}

// The Matrix K is a 3 x 3 matrix with format as follow:
//  |  alpha_x    0     cx |
//  |     0    alpha_y  cy |
//  |     0       0      1 |
//
// Such that the dual omega = KK^T
//      |   alpha_x * alpha_x + cx * cx        cx * cy      cx |
// KK^T =|     cx * cy           alpha_y * alpha_y * cy * cy cy |
//      |        cx                          cy             1  |
//
//  KK^T =   P * Q * P^T
//  We will solve Q with the constraint about the matrix KK^T
//
void ComputeIntrinsicMatrix(std::map<IndexT, Mat34>& projective_reconstruction,
                            size_t image_width, size_t image_height) {
  // cx = image_width / 2
  // cy = image_height / 2
  // alpha_x / alpha y = image_width / image_height (not used)
  // this is the three constraint to solve a linear function.
  //
  // 16 parameter need to 8 photos to compute at least.
  //

  size_t cx = image_width / 2;
  size_t cy = image_width / 2;
  //  4 * projective_reconstruction * 16
  size_t cameras_size = projective_reconstruction.size();
  Eigen::MatrixXd coeffient(4 * cameras_size, 16);
  Eigen::VectorXd constant(4 * cameras_size);
  size_t count = 0;
  for (auto& item_pair : projective_reconstruction) {
    const Mat34& P_i = item_pair.second;
    coeffient.row(count * 4 + 0) = GenerateCoeffient(P_i, 1, 3);
    coeffient.row(count * 4 + 1) = GenerateCoeffient(P_i, 2, 3);
    coeffient.row(count * 4 + 2) = GenerateCoeffient(P_i, 3, 1);
    coeffient.row(count * 4 + 3) = GenerateCoeffient(P_i, 3, 2);
    constant(count * 4 + 0) = cx;
    constant(count * 4 + 1) = cy;
    constant(count * 4 + 2) = cx;
    constant(count * 4 + 3) = cy;
    count++;
  }
  std::cout << "Generate coeffient Finish" << std::endl;

  // Solve Least-Squares Method
  Eigen::VectorXd Q_coeffient = coeffient.colPivHouseholderQr().solve(constant);
  std::cout << "coeffient : " << Q_coeffient << std::endl;
  Eigen::Matrix4d Q =
      Eigen::Map<Eigen::Matrix<double, 4, 4>>(Q_coeffient.data());
  std::cout << "Solved " << Q << std::endl;
  // SVD Q = HIH with I is diag(1, 1, 1, 0)
  Eigen::JacobiSVD svd(Q, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector4d diag = svd.singularValues();
  std::cout << "Singular Value : " << diag << std::endl;
  diag(3) = 0.0;

  Q = svd.matrixU() * diag.asDiagonal() * svd.matrixV();
  std::cout << "Q Matrix : " << Q << std::endl;
  svd = Eigen::JacobiSVD(Q, Eigen::ComputeFullV);
  Eigen::Matrix4d H = svd.matrixV();

  for (auto& item : projective_reconstruction) {
    item.second = item.second * H;
  }
}

void PMatrixToCanonical(const Mat34& P, Eigen::Matrix4d& H) {
  // P * H = [I | 0]
  Eigen::Vector4d C = NullSpace(P);

  H << P, C.transpose();
  H = H.inverse();
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
  }
  // Filter With fundamental matrix
  std::map<Pair, Eigen::Matrix3d> fundamental_matrix;
  std::map<Pair, Matches> fundamental_filter_matches;

  for (const auto& iter : sfm_data.matches) {
    Pair pair = iter.first;
    if (iter.second.size() < 30) {
      continue;
    }
    std::printf("Compute Fundamental Matrix for pair <%lld, %lld>\n",
                pair.first, pair.second);
    const std::vector<KeyPoint>& lhs_keypoints =
        sfm_data.key_points.at(pair.first);
    const std::vector<KeyPoint>& rhs_keypoints =
        sfm_data.key_points.at(pair.second);

    std::vector<KeyPoint> lhs, rhs;
    lhs.reserve(iter.second.size());
    rhs.reserve(iter.second.size());
    for (const Matche& m : iter.second) {
      lhs.push_back(lhs_keypoints[m.first]);
      rhs.push_back(rhs_keypoints[m.second]);
    }

    Eigen::Matrix3d F;
    std::vector<size_t> inlier_index;
    if (ComputeFundamentalMatrix(lhs, rhs, &F, &inlier_index)) {
      fundamental_matrix.insert({pair, F});
      Matches filter_matchs;
      const Matches& origin_matches = iter.second;
      for (size_t index : inlier_index) {
        filter_matchs.push_back(origin_matches.at(index));
      }
    }
  }
  std::printf("%lu Matches are reserved after fundamental matrix filter\n",
              fundamental_matrix.size());

  std::set<IndexT> id_set;
  auto v = fundamental_matrix |
           Transform([](auto item) { return item.first; }) | ToVector();
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
  std::map<IndexT, Mat34> projective_reconstruction;
  ProjectiveReconstruction(fundamental_matrix, fundamental_filter_matches,
                           sfm_data.key_points, projective_reconstruction);

  std::cout << "Reconstruction : " << projective_reconstruction.size()
            << std::endl;
  for (auto iter : projective_reconstruction) {
    std::cout << iter.second << std::endl;
    Eigen::Matrix4d H;
    PMatrixToCanonical(iter.second, H);
    std::cout << "Canonical : " << iter.second * H << std::endl;
  }

  // Compute camera matrix K with the
  // IAC
  ComputeIntrinsicMatrix(projective_reconstruction, 3072, 2304);

  // triangulation
  //
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

  ToPly(sfm_data.structure_points, "./sparse_point.ply");

  // bundle adjustment the parameters rotation and translation

  Save(sfm_data, argv[1]);

  return 0;
}