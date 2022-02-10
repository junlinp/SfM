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
#include "solver/resection_solver.hpp"
bool ComputeFundamentalMatrix(const std::vector<Observation>& lhs_keypoint,
                              const std::vector<Observation>& rhs_keypoint,
                              Mat33* fundamental_matrix,
                              std::vector<size_t>* inlier_index_ptr) {
  assert(lhs_keypoint.size() == rhs_keypoint.size());
  std::vector<typename EightPointFundamentalSolver::DataPointType> datas;
  for (int i = 0; i < lhs_keypoint.size(); i++) {
    Eigen::Vector2d lhs_temp(lhs_keypoint[i].x(), lhs_keypoint[i].y());
    Eigen::Vector2d rhs_temp(rhs_keypoint[i].x(), rhs_keypoint[i].y());
    datas.push_back({lhs_temp, rhs_temp});
  }
  EightPointFundamentalSolver ransac_solver;

  ransac_solver.Fit(datas, *fundamental_matrix);
  double threshold = 3.84;

  std::vector<size_t> inlier_indexs;
  for (int i = 0; i < lhs_keypoint.size(); i++) {
    Eigen::Vector2d lhs_temp(lhs_keypoint[i].x(), lhs_keypoint[i].y());
    Eigen::Vector2d rhs_temp(rhs_keypoint[i].x(), rhs_keypoint[i].y());
    double error =
        //SampsonError::Error({lhs_temp, rhs_temp}, *fundamental_matrix);
        EpipolarLineError::Error({lhs_temp, rhs_temp}, *fundamental_matrix);
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

bool ComputeFundamentalMatrix(const std::vector<KeyPoint>& lhs_keypoint,
                              const std::vector<KeyPoint>& rhs_keypoint,
                              Mat33* fundamental_matrix,
                              std::vector<size_t>* inlier_index_ptr) {
  assert(lhs_keypoint.size() == rhs_keypoint.size());
  std::vector<typename EightPointFundamentalSolver::DataPointType> datas;
  for (int i = 0; i < lhs_keypoint.size(); i++) {
    Eigen::Vector2d lhs_temp(lhs_keypoint[i].x, lhs_keypoint[i].y);
    Eigen::Vector2d rhs_temp(rhs_keypoint[i].x, rhs_keypoint[i].y);
    datas.push_back({lhs_temp, rhs_temp});
  }
  EightPointFundamentalSolver ransac_solver;

  ransac_solver.Fit(datas, *fundamental_matrix);
  double threshold = 3.84;

  std::vector<size_t> inlier_indexs;
  for (int i = 0; i < lhs_keypoint.size(); i++) {
    Eigen::Vector2d lhs_temp(lhs_keypoint[i].x, lhs_keypoint[i].y);
    Eigen::Vector2d rhs_temp(rhs_keypoint[i].x, rhs_keypoint[i].y);
    double error =
        //SampsonError::Error({lhs_temp, rhs_temp}, *fundamental_matrix);
        EpipolarLineError::Error({lhs_temp, rhs_temp}, *fundamental_matrix);
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
  std::vector<typename EightPointFundamentalSolver::DataPointType> datas;
  for (auto&& m : match) {
    datas.push_back({m.lhs_observation, m.rhs_observation});
  }
  EightPointFundamentalSolver ransac_solver;
  ransac_solver.Fit(datas, *fundamental_matrix);
  double threshold = 3.48;
  for (auto&& m : match) {
    //double error = SampsonError::Error({m.lhs_observation, m.rhs_observation},
    //                                   *fundamental_matrix);
    double error = EpipolarLineError::Error({m.lhs_observation, m.rhs_observation}, *fundamental_matrix);
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

bool Redundant(Matches matches) {
  std::set<IndexT> rhs_indexs;
  for (Match m : matches) {
    if (rhs_indexs.find(m.rhs_idx) == rhs_indexs.end()) {
      rhs_indexs.insert(m.rhs_idx);
    } else {
      return true;
    }
  }
  return false;
}
double Error(const Mat34& P1, const Observation& ob1, Eigen::Vector4d& X) {
  Eigen::Vector3d uv = P1 * X;
  return (ob1 - uv.hnormalized()).squaredNorm();
}

double Error(const Mat34& P1,const Observation& ob1, const Mat34& P2, const Observation& ob2, Eigen::Vector4d& X) {
  return (Error(P1, ob1, X) + Error(P2, ob2, X)) / 2.0;
}

void FCheck(Mat33 fundamental_matrix, Matches inliers_matches) {
  Mat34 P1, P2;
 P1 << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  Eigen::Vector3d e12 = NullSpace(fundamental_matrix.transpose().eval());
  //std::cout << fundamental_matrix.transpose() * e12 << std::endl;
  P2 << SkewMatrix(e12) * fundamental_matrix, e12;

  for (Match match : inliers_matches) {
    //std::cout << "Match Compute : " << std::endl << std::endl;

    //std::cout << "lhs observation :" << match.lhs_observation << std::endl;
    //std::cout << "rhs observation : " << match.rhs_observation << std::endl;
    //std::cout << "P1 : " << P1 << std::endl;
    //std::cout << "P2 : " << P2 << std::endl;


    //std::cout << "x' * F * x : " << match.rhs_observation.homogeneous().dot(fundamental_matrix * match.lhs_observation.homogeneous()) << std::endl;
    Eigen::Vector3d epipolar_line_2 = fundamental_matrix * match.lhs_observation.homogeneous();
    Eigen::Vector3d epipolar_line_1 = fundamental_matrix.transpose() * match.rhs_observation.homogeneous();
    auto squared = [](auto num) { return num * num;};
    //std::cout << "Line2 square error : " << squared(epipolar_line_2.dot(match.rhs_observation.homogeneous())) / (epipolar_line_2.x()*epipolar_line_2.x() + epipolar_line_2.y() * epipolar_line_2.y())<< std::endl;
    //std::cout << "Line1 square error : " << squared(epipolar_line_1.dot(match.lhs_observation.homogeneous())) / (epipolar_line_1.x()*epipolar_line_1.x() + epipolar_line_1.y() * epipolar_line_1.y())<< std::endl;

    //std::cout << "Line2 error : " << (epipolar_line_2.dot(match.rhs_observation.homogeneous())) / sqrt(epipolar_line_2.x()*epipolar_line_2.x() + epipolar_line_2.y() * epipolar_line_2.y())<< std::endl;
    //std::cout << "Line1 error : " << (epipolar_line_1.dot(match.lhs_observation.homogeneous())) / sqrt(epipolar_line_1.x()*epipolar_line_1.x() + epipolar_line_1.y() * epipolar_line_1.y())<< std::endl;

    Eigen::Vector3d b_ = P2.block(0, 0, 3, 3) * match.lhs_observation.homogeneous();
    Eigen::MatrixXd A(3, 2);
    A.col(0) = match.rhs_observation.homogeneous();
    A.col(1) = -P2.col(3);
    Eigen::Vector2d solution = (A.transpose() * A).inverse() * A.transpose() * b_;
    //std::cout << "depth : " << solution(1) << std::endl;
    Eigen::Vector4d X = (match.lhs_observation.homogeneous()).homogeneous();
    X(3) *= solution(1);
    
    Eigen::Vector3d uv = P2 * X;
    Eigen::Vector3d l = SkewMatrix(e12) * uv;
    
    //std::cout << "l : " << l << std::endl;
    //std::cout << "l dot t : " << l.dot(match.rhs_observation.homogeneous()) << std::endl;
    //std::cout << "sqrt(A**2 + B**2) : " << l(0) * l(0) + l(1) * l(1) << std::endl;
    //std::cout << "Estimate Line error : " << squared(l.dot(match.rhs_observation.homogeneous())) / (l(0) * l(0) + l(1) * l(1)) << std::endl;
    const Observation & rhs_ob = match.rhs_observation;
    double a = l.x();
    double b = l.y();
    double c = l.z();
    double c_ = b * rhs_ob.x() - a * rhs_ob.y();
    double y = (-c_ - c * b / a) * a / (a*a+b*b);
    double x = (-c - b * y) / a;
    Eigen::Vector4d dlt_X;
    DLT({P1, P2}, {match.lhs_observation.homogeneous(), match.rhs_observation.homogeneous()}, dlt_X);
    if (Error(P1, match.lhs_observation, P2, match.rhs_observation, dlt_X) > 3.84) {
      std::cout << "Geometry Error : " <<  Error(P1, match.lhs_observation, P2, match.rhs_observation, dlt_X) << std::endl;
      std::cout << "X : " << X / X(3) << std::endl;
      std::cout << "The Point : " << x << ", " << y << std::endl;
      std::cout << "uv : " << uv.hnormalized() << std::endl;
      std::cout << "ob : " << match.rhs_observation << std::endl;
      std::cout << "P1 * X : " << (P1 * X).hnormalized() << std::endl;
      std::cout << "ob1 : " << match.lhs_observation << std::endl;
      std::cout << "P2 * X : " << (P2 * X).hnormalized() << std::endl;
      std::cout << "ob2 : " << match.rhs_observation << std::endl;
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
  std::cout << "Fundamental Matrix singular value : " << fundamental_matrix.jacobiSvd().singularValues() << std::endl;
  //FCheck(fundamental_matrix, inliers_matches);
  //std::cout << "Redundant : " << Redundant(inliers_matches) << std::endl;

  // the fundamental matrix error is 1e-3
  // but the DLT will be so large
  
  std::printf("%lu inliers matches\n", inliers_matches.size());
  // return Structure struct
  Mat34 P1, P2;
  // P1 = [I | 0]
  // P2 = [ [e12]_x * F_12 + e12 * a^T | sigma * e12]
  // we using a^T = (0, 0, 0) and sigma = 1.0
  P1 << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  Eigen::Vector3d e12 = NullSpace(fundamental_matrix.transpose().eval());
  std::cout << fundamental_matrix.transpose() * e12 << std::endl;
  P2 << SkewMatrix(e12) * fundamental_matrix, e12;
  
  ProjectiveStructure projective_structure(*sfm_data);

  projective_structure.InitializeStructure(initial_pair, inliers_matches, P1,
                                           P2);
  return projective_structure;
}

// c. Find the next image to register
bool ProjectiveReconstruction(ProjectiveStructure& structure) {
  IndexT image_id = -1;
  int register_count = 0;
  while ((image_id = structure.NextImage()) != -1) {
    Correspondence cor = structure.FindCorrespondence(image_id);
    std::printf("Next Image %lld with correspondence %lu\n", image_id, cor.size());
    if (cor.size() < 10) {
      structure.UnRegister(image_id);
    } else {
      std::printf("Register Imaage %lld\n", image_id);
      register_count++;
      //std::vector<Observation> observation_2d = cor.Getobservations();
      //std::vector<Eigen::Vector3d> points = cor.GetPoints();
      
      auto&& _2d_3d_cor = cor.Get2D_3D();
      std::vector<size_t> inliers;
      Mat34 P;
      // TODO (junlinp@qq.com):
      // ransac DLT
      //DLT(observation_2d, points, P);
      RansacResection resection_solver;
      bool b_resection = resection_solver.Resection(_2d_3d_cor, P, &inliers);
      if (!b_resection) {
        std::printf("Resection Fails\n");
        continue;
      } else {
        std::printf("Correspondence Inlier %lu\n", inliers.size());
        cor = cor.Filter(inliers);
      } 

      structure.Register(image_id, P, cor);

      // g. Refine Local Structure
      structure.LocalBundleAdjustment();

      // f. Triangularing new sparse point.
      structure.TriangularNewPoint(image_id);

    }
  }
  return register_count > 0;
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


struct MatchesWrap {
  std::map<Pair, Matches> matches;
  MatchesWrap(std::map<Pair, Matches> matches) : matches(matches) {}

  bool Exists(Pair pair) const {
    auto origin_iterator = matches.find(pair);
    auto reverse_iterator = matches.find({pair.second, pair.first});

    return origin_iterator != matches.end() || reverse_iterator != matches.end();
  }

  Matches at(Pair pair) const {

    if (matches.find(pair) != matches.end()) {
      return matches.at(pair);
    }

    if (matches.find({pair.second, pair.first}) != matches.end()) {
      Matches res;
      const Matches& reverse = matches.at({pair.second, pair.first});
      for (Match match : reverse) {
        std::swap(match.lhs_observation, match.rhs_observation);
        std::swap(match.lhs_idx, match.rhs_idx);
        res.push_back(match);
      }
      return res;
    }

    return Matches{};
  }
};


std::vector<TripleIndex> ListTriple(const SfMData& sfm_data) {
  std::vector<TripleIndex> res;
  std::vector<IndexT> total_index = sfm_data.views | Transform([](auto pair){return pair.first;}) | ToVector();
  size_t max_size = total_index.size();

  for(size_t i = 0; i < max_size; i++) {
    for (size_t j = i + 1; j < max_size; j++) {
      for (size_t k = j + 1; k < max_size; k++) {
        res.push_back(TripleIndex{total_index[i], total_index[j], total_index[k]});
      }
    }
  }
  return res;
}

bool ValidateTriple(TripleIndex index,const MatchesWrap& match) {
  return match.Exists({index.I, index.J}) && match.Exists({index.J, index.K}) && match.Exists({index.K, index.I});
}

std::vector<TripleMatch> ConstructTripleMatch(TripleIndex index, const MatchesWrap& match) {

  Matches I_J_matches = match.at({index.I, index.J});
  Matches J_K_matches = match.at({index.J, index.K});
  Matches K_I_matches = match.at({index.K, index.I});
  
  std::map<IndexT, Match> j_match_k;
  std::map<IndexT, Match> k_match_i;
  for (Match j_k : J_K_matches) {
    j_match_k[j_k.lhs_idx] = j_k;
  }

  for (Match k_i : K_I_matches) {
    k_match_i[k_i.lhs_idx] = k_i;
  }
  std::vector<TripleMatch> res;

  for (Match i_j : I_J_matches) {
    // J should has match to k
    IndexT j_index = i_j.rhs_idx;
    if (j_match_k.find(j_index) != j_match_k.end()) {
      Match j_k = j_match_k[j_index];
      IndexT k_index = j_k.rhs_idx;

      if (k_match_i.find(k_index) != k_match_i.end()) {
        Match k_i = k_match_i.at(k_index);
        IndexT i_index = k_i.rhs_idx;

        if (i_j.lhs_idx == i_index) {
          TripleMatch m;
          m.I_idx = i_index; m.J_idx = j_index; m.K_idx = k_index;
          m.I_observation = i_j.lhs_observation;
          m.J_observation = i_j.rhs_observation;
          m.K_observation = j_k.rhs_observation;
          res.push_back(m);
        }
      }
    }



  }
  return res;
}

std::pair<TripleIndex, std::vector<TripleMatch>> FindInitialTriple(const SfMData& sfm_data) {
  
  MatchesWrap match(sfm_data.matches);

  auto temp = ListTriple(sfm_data) | Transform([match](auto triple_index) { return std::pair<TripleIndex, std::vector<TripleMatch>>{triple_index, ConstructTripleMatch(triple_index, match)};}) | ToVector();
  
  std::sort(temp.begin(), temp.end(),[](auto lhs_pair, auto rhs_pair) { return lhs_pair.second.size() > rhs_pair.second.size();});
  return temp[0];
}

std::ostream& operator<<(std::ostream& os, TripleIndex index) {
  os << "[" << index.I << "," << index.J << "," << index.K << "]";
  return os;
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

  auto triple_ = FindInitialTriple(sfm_data);
  std::cout << triple_.first << " : " << triple_.second.size() << std::endl; 
  std::cout << sfm_data.matches.size() << "Match " << std::endl;

  LinearSolver trifocal_solver;
  Trifocal trifocal;
  trifocal_solver.Fit(triple_.second, trifocal);
  Mat34 P1, P2, P3;
  RecoveryCameraMatrix(trifocal, P1, P2, P3);
  std::cout << "P1 : " << P1 << std::endl;
  std::cout << "P2 : " << P2 << std::endl;
  std::cout << "P3 : " << P3 << std::endl;

  std::map<Pair, Matches> filter_matches;

  for (auto&& [pair, matches] : sfm_data.matches) {
    std::vector<Observation> lhs, rhs;
    for (Match match : matches) {
      lhs.push_back(match.lhs_observation);
      rhs.push_back(match.rhs_observation);
    }

    Mat33 fundamental;
    std::vector<size_t> inlier;
    bool ans = ComputeFundamentalMatrix(lhs, rhs, &fundamental, &inlier);
    if (ans) {

      Matches t;
      for (size_t index : inlier) {
        t.push_back(matches[index]);
      }
      filter_matches[pair] = t;
    }
  }

  sfm_data.matches = filter_matches;

  std::cout << "After Filter : " << sfm_data.matches.size() << "Match " << std::endl;

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

  projective_structure.LocalBundleAdjustment();

  auto PS = projective_structure.GetPMatrix();
  Eigen::Matrix4d Q = IAC(PS, sfm_data.image_width, sfm_data.image_height);

  // TODO: junlinp@qq.com
  // the eigen value of Q is not identity.
  // 
  auto svd = Q.bdcSvd();
  std::cout << "Q singularValue : " << svd.singularValues() << std::endl;
  projective_structure.ApplyQMatrix(Q);

  // IAC to Compute K R C and the homogeneous transform of Q
  // to a True Euclidean Reconstruction

  std::vector<Track> tracks = projective_structure.GetTracks();
  for (Track track : tracks) {
    std::cout << "Point : " << track.X << std::endl;
    double x = track.X.x();
    double y = track.X.y();
    double z = track.X.z();
    if (std::abs(x) < 1000.0 && std::abs(y) < 1000.0 && std::abs(z) < 1000.0) {
      SparsePoint sp(track.X.x(), track.X.y(), track.X.z());
      sfm_data.structure_points.push_back(sp);
    }
  }
  std::printf("%lu Point totally\n", tracks.size());
  ToPly(sfm_data.structure_points, "./sparse_point.ply");
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



  // bundle adjustment the parameters rotation and translation

  Save(sfm_data, argv[1]);

  return 0;
}