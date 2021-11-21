#include <fstream>
#include <iostream>
#include <queue>

#include "Eigen/Dense"
#include "internal/function_programming.hpp"
#include "ransac.hpp"
#include "sfm_data.hpp"
#include "sfm_data_io.hpp"
#include "solver/algebra.hpp"
#include "solver/fundamental_solver.hpp"
#include "solver/self_calibration_solver.hpp"
#include "solver/triangular_solver.hpp"
#include "solver/trifocal_tensor_solver.hpp"

// Union Find Set to construct track
// Define Hash and UnHash Function

int64_t Hash(int32_t image_idx, int32_t feature_idx) {

  return ((int64_t)image_idx << 32) + (int64_t)(feature_idx);
}

void UnHash(int64_t hash_value, int32_t& image_idx, int32_t& feature_idx) {
  feature_idx = hash_value & 0xffffffff;
  image_idx = (hash_value >> 32) & 0xffffffff;
}

template<typename IndexType>
class UnionFindSet {
  public:
    void insertIdx(IndexType idx) {
      idx_to_parent[idx] = idx;
    }

    IndexType FindRoot(IndexType idx) {
      while(idx_to_parent[idx] != idx) {
        idx = idx_to_parent[idx];
      }
      return idx;
    }

    void Union(IndexType lhs_idx, IndexType rhs_idx) {
      IndexType lhs_root = FindRoot(lhs_idx);
      IndexType rhs_root = FindRoot(rhs_idx);
      idx_to_parent[lhs_root] = rhs_root;
    }

    IndexType DifferentSetSize() {
      std::set<IndexType> roots;
      for(auto pair : idx_to_parent) {
        auto root = FindRoot(pair.first);
        roots.insert(root);
        idx_to_parent[pair.first] = root;
      }
      return roots.size();
    }

  private:

  std::map<IndexType, IndexType> idx_to_parent;
};

//
void BuildTrack(std::map<IndexT, std::vector<KeyPoint>> key_points, std::map<Pair, Matches> matches) {
  UnionFindSet<int64_t> ufs;

  for (auto pair : key_points) {
    int32_t image_idx = pair.first;
    const std::vector<KeyPoint>& key_point = pair.second;
    for(int32_t i = 0; i < key_point.size(); i++) {
      int64_t idx = Hash(image_idx, i);
      ufs.insertIdx(idx);
    }
  }

  // Union Find the matches
  for(auto match_pair : matches) {
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

std::vector<TriPair> TripleTrackBuilder(const std::vector<KeyPoint>& I_keypoint,
                                        const std::vector<KeyPoint>& J_keypoint,
                                        const std::vector<KeyPoint>& K_keypoint,
                                        const Matches& I_to_J_matches,
                                        const Matches& J_to_K_matches,
                                        const Matches& I_to_K_matches) {
  std::vector<TriPair> res;

  std::map<IndexT, IndexT> J_to_K_matches_map;
  std::map<IndexT, IndexT> K_to_I_matches_map;
  // I -> J -> K -> I
  for (Matche match : J_to_K_matches) {
    J_to_K_matches_map[match.first] = match.second;
  }

  for (Matche match : I_to_K_matches) {
    K_to_I_matches_map[match.second] = match.first;
  }

  for (Matche match : I_to_J_matches) {
    if (J_to_K_matches_map.find(match.second) != J_to_K_matches_map.end()) {
      IndexT k = J_to_K_matches_map[match.second];
      if (K_to_I_matches_map.find(k) != K_to_I_matches_map.end() &&
          K_to_I_matches_map[k] == match.first) {
        Eigen::Vector2d I_obs(I_keypoint[match.first].x,
                              I_keypoint[match.first].y);
        Eigen::Vector2d J_obs(J_keypoint[match.second].x,
                              J_keypoint[match.second].y);
        Eigen::Vector2d K_obs(K_keypoint[k].x, K_keypoint[k].y);
        TriPair p(I_obs, J_obs, K_obs);
        res.push_back(p);
      }
    }
  }
  return res;
};

void ProjectiveReconstruction(
    const std::map<Pair, Eigen::Matrix3d>& fundamental_matrix,
    const std::map<Pair, Matches>& filter_matches,
    const std::map<IndexT, std::vector<KeyPoint>>& keypoint,
    std::map<IndexT, Mat34>& projective_reconstruction,
    std::vector<SparsePoint>& sparse_point_vec

) {
  using Fundamental_Matrix_Type =
      typename std::map<Pair, Eigen::Matrix3d>::value_type;

  // A. find all the triple using the fundamental_matrix.
  // B. find a best triple to initialize.
  // C. Init three camera using trifocal.
  // D. tranverse the rest of triple which two of that has been initilized.
  // E. go to D until all tirple is processed.

  // A. find all the triple using the fundamental_matrix
  std::printf("A. find all the triple using the fundamental matrix\n");
  std::set<IndexT> index_set;
  auto all_index = fundamental_matrix |
                   Transform([](const auto& item) { return item.first; }) |
                   ToVector();
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
                fundamental_matrix.find(two) != fundamental_matrix.end()) {
              TripleIndex triple{I, J, K};
              triple_pair.push_back(triple);
            }
          }
        }
      }
    }
  }
  for (TripleIndex lhs : triple_pair) {
    auto lhs_data_point = TripleTrackBuilder(
        keypoint.at(lhs.I_), keypoint.at(lhs.J_), keypoint.at(lhs.K_),
        filter_matches.at({lhs.I_, lhs.J_}),
        filter_matches.at({lhs.J_, lhs.K_}),
        filter_matches.at({lhs.I_, lhs.K_}));
    std::cout << "Triple Feature : " << lhs_data_point.size() << std::endl;
  }

  std::sort(triple_pair.begin(), triple_pair.end(),
            [keypoint, filter_matches](const TripleIndex& lhs,
                                       const TripleIndex& rhs) {
              auto lhs_data_point = TripleTrackBuilder(
                  keypoint.at(lhs.I_), keypoint.at(lhs.J_), keypoint.at(lhs.K_),
                  filter_matches.at({lhs.I_, lhs.J_}),
                  filter_matches.at({lhs.J_, lhs.K_}),
                  filter_matches.at({lhs.I_, lhs.K_}));
              auto rhs_data_point = TripleTrackBuilder(
                  keypoint.at(rhs.I_), keypoint.at(rhs.J_), keypoint.at(rhs.K_),
                  filter_matches.at({rhs.I_, rhs.J_}),
                  filter_matches.at({rhs.J_, rhs.K_}),
                  filter_matches.at({rhs.I_, rhs.K_}));
              return lhs_data_point.size() > rhs_data_point.size();
            });
  // B.find a best triple to initialize.
  std::printf("B. find a best triple to initialize\n");
  std::set<IndexT> used_index;
  TripleIndex initial = triple_pair.front();
  used_index.insert(initial.I_);
  used_index.insert(initial.J_);
  used_index.insert(initial.K_);

  {
    const std::vector<KeyPoint>& I_keypoint = keypoint.at(initial.I_);
    const std::vector<KeyPoint>& J_keypoint = keypoint.at(initial.J_);
    const std::vector<KeyPoint>& K_keypoint = keypoint.at(initial.K_);

    std::vector<TriPair> data_points =
        TripleTrackBuilder(I_keypoint, J_keypoint, K_keypoint,
                           filter_matches.at({initial.I_, initial.J_}),
                           filter_matches.at({initial.J_, initial.K_}),
                           filter_matches.at({initial.I_, initial.K_}));
    if (data_points.size() < 7) {
      std::printf("Initial Pair fails\n");
      return;
    }
    Trifocal trifocal;
    BundleRefineSolver solver;
    solver.Fit(data_points, trifocal);
    RecoveryCameraMatrix(trifocal, projective_reconstruction[initial.I_],
                         projective_reconstruction[initial.J_],
                         projective_reconstruction[initial.K_]);
    for (TriPair item : data_points) {
      // DLT
      Eigen::Vector4d X;
      DLT({projective_reconstruction[initial.I_],
           projective_reconstruction[initial.J_],
           projective_reconstruction[initial.K_]},
          {item.lhs, item.middle, item.rhs}, X);
      Eigen::Vector3d x = X.hnormalized();
      SparsePoint po(x.x(), x.y(), x.z());
      po.obs[initial.I_] = item.lhs.hnormalized();
      po.obs[initial.J_] = item.middle.hnormalized();
      po.obs[initial.K_] = item.rhs.hnormalized();
      sparse_point_vec.push_back(po);
    }
  }

  auto ContanerExist = [](auto container, auto item) {
    return container.find(item) != container.end();
  };

  auto TripleValid = [ContanerExist](const TripleIndex& triple_index,
                                     const std::set<IndexT>& used_index) {
    int count = 0;
    if (ContanerExist(used_index, triple_index.I_)) {
      count++;
    }

    if (ContanerExist(used_index, triple_index.J_)) {
      count++;
    }
    if (ContanerExist(used_index, triple_index.K_)) {
      count++;
    }
    return count == 2;
  };
  // remove the initial pair
  triple_pair.erase(triple_pair.begin());
  bool found = true;
  while (found) {
    found = false;

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
        const std::vector<KeyPoint>& first_keypoint = keypoint.at(first);
        const std::vector<KeyPoint>& second_keypoint = keypoint.at(second);
        const std::vector<KeyPoint>& need_to_predict_keypoint =
            keypoint.at(need_to_predict);
        auto GetCorrectMatchs = [filter_matches](size_t i, size_t j) {
          auto res = filter_matches.at({std::min(i, j), std::max(i, j)});
          if (j > i) {
            for (auto& item : res) {
              std::swap(item.first, item.second);
            }
          }
          return res;
        };
        std::vector<TriPair> data_point = TripleTrackBuilder(
            first_keypoint, second_keypoint, need_to_predict_keypoint,
            GetCorrectMatchs(first, second),
            GetCorrectMatchs(second, need_to_predict),
            GetCorrectMatchs(need_to_predict, first));
        if (data_point.size() < 7) {
          continue;
        }

        Trifocal trifocal;
        BundleRefineSolver solver;
        solver.Fit(data_point, trifocal);

        BundleRecovertyCameraMatrix(data_point, trifocal,
                                    projective_reconstruction[first],
                                    projective_reconstruction[second],
                                    projective_reconstruction[need_to_predict]);
        for (TriPair item : data_point) {
          Eigen::Vector4d X;
          DLT({projective_reconstruction[first],
               projective_reconstruction[second],
               projective_reconstruction[need_to_predict]},
              {item.lhs, item.middle, item.rhs}, X);
          Eigen::Vector3d x = X.hnormalized();
          // DLT
          SparsePoint po;
          po.obs[first] = item.lhs.hnormalized();
          po.obs[second] = item.middle.hnormalized();
          po.obs[need_to_predict] = item.rhs.hnormalized();
          sparse_point_vec.push_back(po);
        }
        // append
        found = true;
        used_index.insert(triple_index.I_);
        used_index.insert(triple_index.J_);
        used_index.insert(triple_index.K_);
      }
      // TODO: global bundle adjustment after adding a new camera.
    }
    std::printf("Processed %lu Cameras\n", used_index.size());
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

int intesection(const Matches& I_J, const Matches& J_K) {
  std::set<IndexT> I_J_set;
  std::set<IndexT> J_K_set;
  for (auto m : I_J) {
    I_J_set.insert(m.second);
  }

  for (auto m : J_K) {
    J_K_set.insert(m.first);
  }

  int ans = 0;
  for (IndexT index : I_J_set) {
    if (J_K_set.find(index) != J_K_set.end()) {
      ans++;
    }
  }
  std::cout << "I_J : " << I_J.size() << " : "
            << " J_K : " << J_K.size() << std::endl;
  return ans;
}

int intesection(const Matches& I_J, const Matches& J_K, const Matches& K_I) {
  std::map<IndexT, IndexT> J_K_map;
  std::map<IndexT, IndexT> K_I_map;
  for (auto m : J_K) {
    J_K_map[m.first] = m.second;
  }

  for (auto m : K_I_map) {
    K_I_map[m.first] = m.second;
  }
  int ans = 0;
  for (auto m : I_J) {
    IndexT I = m.first;
    IndexT J = m.second;

    if (J_K_map.find(J) != J_K_map.end()) {
      IndexT K = J_K_map[J];
      if (K_I_map.find(K) != K_I_map.end() && K_I_map[K] == I) {
        ans++;
      }
    }
  }
  return ans;
}

struct Triple {
  IndexT i_, j_, k_;

  bool operator<(const Triple& rhs) const {
    if (i_ == rhs.i_) {
      if (j_ == rhs.j_) {
        return k_ < rhs.k_;
      }
      return j_ < rhs.j_;
    }
    return i_ < rhs.i_;
  }
};
std::vector<Triple> GenerateTriple(const Matches& I_J, const Matches& J_K) {
  std::map<IndexT, IndexT> J_K_map;
  std::map<IndexT, IndexT> J_I_map;

  for (auto m : J_K) {
    J_K_map.insert({m.first, m.second});
  }

  for (auto m : I_J) {
    J_I_map[m.second] = m.first;
  }
  std::set<Triple> filter_set;
  std::vector<Triple> ans;
  for (auto index : I_J) {
    if (J_K_map.find(index.second) != J_K_map.end()) {
      IndexT i = index.first;
      IndexT j = index.second;
      IndexT k = J_K_map.at(index.second);
      filter_set.insert({i, j, k});
    }
  }

  for (auto index : J_K) {
    if (J_I_map.find(index.first) != J_I_map.end()) {
      IndexT i = J_I_map.at(index.first);
      IndexT j = index.first;
      IndexT k = index.second;
      filter_set.insert({i, j, k});
    }
  }
  for(auto item : filter_set) {
    ans.push_back(item);
  }
  return ans;
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
  BuildTrack(sfm_data.key_points, sfm_data.matches);
  return 0;
  // Filter With fundamental matrix
  std::map<Pair, Eigen::Matrix3d> fundamental_matrix;
  std::map<Pair, Matches> fundamental_filter_matches;
  std::cout << intesection(sfm_data.matches.at({0, 7}),
                           sfm_data.matches.at({7, 8}))
            << std::endl;
  std::cout << intesection(sfm_data.matches.at({0, 8}),
                           ReverseMatches(sfm_data.matches.at({7, 8})))
            << std::endl;
  std::cout << intesection(sfm_data.matches.at({0, 7}),
                           sfm_data.matches.at({7, 8}),
                           ReverseMatches(sfm_data.matches.at({0, 8})))
            << std::endl;
  auto res = GenerateTriple(sfm_data.matches.at({0, 7}),
                           sfm_data.matches.at({7, 8}));
  std::cout << res.size() << std::endl;

  return 0;
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
      fundamental_filter_matches.insert({pair, origin_matches});
      fundamental_filter_matches.insert(
          {{pair.second, pair.first}, ReverseMatches(origin_matches)});
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
  std::printf("Projective Reconstruction\n");
  std::map<IndexT, Mat34> projective_reconstruction;
  ProjectiveReconstruction(fundamental_matrix, fundamental_filter_matches,
                           sfm_data.key_points, projective_reconstruction,
                           sfm_data.structure_points);

  std::cout << "Reconstruction : " << projective_reconstruction.size()
            << std::endl;

  // Compute camera matrix K with the
  // IAC
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